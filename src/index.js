import { loadConfig } from "./config.js"
import { buildPatternSet } from "./patterns.js"
import { PlaceholderSession } from "./session.js"
import { redactText, redactTextWithAI } from "./engine.js"
import { redactDeep, restoreDeep } from "./deep.js"
import { restoreText } from "./restore.js"

/**
 * OpenCode plugin entry point:
 * - `experimental.chat.messages.transform`: redact all messages before sending to LLM
 * - `tool.execute.before`: restore placeholders before local tool execution
 * - `experimental.text.complete`: restore placeholders in completed model output
 *
 * AI detection is opt-in via the `ai` config section. When AI is enabled but
 * @huggingface/transformers is not installed, falls back to regex/keyword only.
 */
export const VibeGuardPrivacy = async (ctx) => {
  const config = await loadConfig(ctx.directory)
  const debug = Boolean(process.env.OPENCODE_VIBEGUARD_DEBUG) || Boolean(config.debug)

  if (debug) {
    const from = config.loadedFrom ? config.loadedFrom : "not found (plugin will no-op)"
    console.log(`[vibeguard] Config: ${from} enabled=${config.enabled}`)
  }

  if (!config.enabled) return {}

  const patterns = buildPatternSet(config.patterns)
  const sessions = new Map()
  const aiConfig = config.ai
  const useAI = aiConfig.enabled

  // Check AI availability at startup (non-blocking info)
  // Import ai-detect lazily to avoid pulling in Transformers.js when AI disabled
  if (useAI) {
    const { isAIAvailable, disposeAI } = await import("./ai-detect.js")
    const available = await isAIAvailable()
    if (available) {
      console.log(
        `[vibeguard] AI detection enabled (model: ${aiConfig.model}, dtype: ${aiConfig.dtype}). ` +
          `Model will be downloaded on first use if not cached.`
      )
    } else {
      console.log(
        `[vibeguard] AI detection enabled in config but @huggingface/transformers is not installed. ` +
          `Install with: npm i @huggingface/transformers\n` +
          `Falling back to regex/keyword detection only.`
      )
    }

    // Clean up model pipeline on process exit to free memory
    const onExit = () => {
      disposeAI().catch(() => {})
    }
    process.on("exit", onExit)
    process.on("SIGINT", onExit)
    process.on("SIGTERM", onExit)
  }

  if (debug) {
    console.log(`[vibeguard] AI detection: ${useAI ? "enabled" : "disabled (opt-in via config)"}`)
    console.log(
      `[vibeguard] Regex patterns: ${patterns.keywords.length} keywords, ${patterns.regex.length} regex rules`
    )
  }

  const getSession = (sessionID) => {
    const key = String(sessionID ?? "")
    if (!key) return null
    const existing = sessions.get(key)
    if (existing) return existing
    const created = new PlaceholderSession({
      prefix: config.prefix,
      ttlMs: config.ttlMs,
      maxMappings: config.maxMappings,
    })
    sessions.set(key, created)
    return created
  }

  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      const msgs = output?.messages
      if (!Array.isArray(msgs) || msgs.length === 0) return

      const sessionID = msgs[0]?.info?.sessionID ?? msgs[0]?.parts?.[0]?.sessionID
      const session = getSession(sessionID)
      if (!session) return

      session.cleanup()

      let changedTextParts = 0

      // Choose redaction function based on AI config
      const redactStr = useAI
        ? async (text) => {
            const result = await redactTextWithAI(text, patterns, session, aiConfig, debug)
            return result.text
          }
        : (text) => {
            return Promise.resolve(redactText(text, patterns, session).text)
          }

      for (const msg of msgs) {
        const parts = Array.isArray(msg?.parts) ? msg.parts : []
        for (const part of parts) {
          if (!part) continue

          // Plain text (user/assistant)
          if (part.type === "text") {
            if (part.ignored) continue
            if (!part.text || typeof part.text !== "string") continue
            const before = part.text
            const after = await redactStr(before)
            if (after !== before) changedTextParts++
            part.text = after
            continue
          }

          // Reasoning text
          if (part.type === "reasoning") {
            if (!part.text || typeof part.text !== "string") continue
            const before = part.text
            const after = await redactStr(before)
            if (after !== before) changedTextParts++
            part.text = after
            continue
          }

          // Tool calls/outputs: most common leak source (e.g., reading .env)
          if (part.type === "tool") {
            const state = part.state
            if (!state || typeof state !== "object") continue

            // Deep-redact tool inputs (args) so they don't leak in later turns.
            // Uses sync regex-only for deep object traversal; AI layer covers
            // text parts and tool output strings.
            if (state.input && typeof state.input === "object") {
              redactDeep(state.input, patterns, session)
            }

            if (state.status === "completed" && typeof state.output === "string") {
              const before = state.output
              const after = await redactStr(before)
              if (after !== before) changedTextParts++
              state.output = after
              continue
            }
            if (state.status === "error" && typeof state.error === "string") {
              const before = state.error
              const after = await redactStr(before)
              if (after !== before) changedTextParts++
              state.error = after
              continue
            }
            if (state.status === "pending" && typeof state.raw === "string") {
              const before = state.raw
              const after = await redactStr(before)
              if (after !== before) changedTextParts++
              state.raw = after
              continue
            }
          }
        }
      }

      if (debug && changedTextParts > 0) {
        console.log(
          `[vibeguard] Pre-request redaction: modified ${changedTextParts} text segment(s)`
        )
      }
    },

    "experimental.text.complete": async (input, output) => {
      if (!output || typeof output !== "object") return
      if (typeof output.text !== "string" || !output.text) return
      const session = getSession(input?.sessionID)
      if (!session) return
      session.cleanup()
      const before = output.text
      const after = restoreText(before, session)
      output.text = after
      if (debug && after !== before) {
        console.log("[vibeguard] Post-response restore: modified 1 text segment")
      }
    },

    "tool.execute.before": async (input, output) => {
      const session = getSession(input?.sessionID)
      if (!session) return
      session.cleanup()
      restoreDeep(output?.args, session)
    },
  }
}
