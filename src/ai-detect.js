/**
 * AI-based PII/secret detection using OpenAI's Privacy Filter model
 * via Transformers.js (runs locally, no external API calls).
 *
 * This module is opt-in: it only activates when `ai.enabled` is set in config
 * AND `@huggingface/transformers` is installed.
 *
 * The model (~400MB q4 quantized) is downloaded on first use and cached locally.
 */

/** Default timeout for model download/load (2 minutes). */
const MODEL_LOAD_TIMEOUT_MS = 120_000

/** Cooldown after a failed model load before retrying (5 minutes). */
const RETRY_COOLDOWN_MS = 5 * 60_000

/** Map Privacy Filter entity labels to vibeguard categories. */
const LABEL_TO_CATEGORY = {
  private_person: "PRIVATE_PERSON",
  private_address: "PRIVATE_ADDRESS",
  private_email: "PRIVATE_EMAIL",
  private_phone: "PRIVATE_PHONE",
  private_url: "PRIVATE_URL",
  private_date: "PRIVATE_DATE",
  account_number: "ACCOUNT_NUMBER",
  secret: "SECRET",
}

let _pipeline = null
let _loading = null
let _transformersModule = undefined // undefined = not yet checked
let _transformersImportFailed = false // true = last import() threw
let _pipelineFailedAt = 0 // timestamp of last pipeline failure (0 = never)

/**
 * Attempt to import @huggingface/transformers.
 * Returns the module or null if not installed.
 * Allows retry: a transient failure (e.g. temporary file-system error)
 * clears the cached null so the next call re-attempts the import.
 */
async function loadTransformers() {
  if (_transformersModule !== undefined && _transformersModule !== null) return _transformersModule
  // If the previous import failed, allow retry (don't cache null permanently)
  if (_transformersModule === null && !_transformersImportFailed) return null
  try {
    _transformersModule = await import("@huggingface/transformers")
    _transformersImportFailed = false
    return _transformersModule
  } catch {
    _transformersModule = null
    _transformersImportFailed = true
    return null
  }
}

/**
 * Initialize the Privacy Filter pipeline. Lazy-loads on first call.
 * Returns the pipeline instance or null if unavailable.
 *
 * Defenses:
 * - If the pipeline loaded successfully before, returns cached instance.
 * - If another caller is already loading, coalesces onto that promise.
 * - If the previous load failed, enforces a cooldown (RETRY_COOLDOWN_MS)
 *   before retrying to avoid a retry storm on every redaction call.
 * - Wraps model download in a timeout (MODEL_LOAD_TIMEOUT_MS) so a
 *   hanging download doesn't block redaction forever.
 */
async function getPipeline(aiConfig, debug) {
  if (_pipeline) return _pipeline
  if (_loading) return _loading

  // Cooldown after failure — don't retry on every call
  if (_pipelineFailedAt > 0) {
    const elapsed = Date.now() - _pipelineFailedAt
    if (elapsed < RETRY_COOLDOWN_MS) {
      if (debug) {
        const remaining = Math.ceil((RETRY_COOLDOWN_MS - elapsed) / 1000)
        console.log(`[vibeguard] AI model load on cooldown (${remaining}s remaining), skipping`)
      }
      return null
    }
    // Cooldown expired — allow retry
    _pipelineFailedAt = 0
  }

  _loading = (async () => {
    const transformers = await loadTransformers()
    if (!transformers) {
      if (debug) {
        console.log(
          "[vibeguard] AI detection unavailable: @huggingface/transformers not installed. " +
            "Install with: npm i @huggingface/transformers"
        )
      }
      return null
    }

    const model = aiConfig.model || "openai/privacy-filter"
    const dtype = aiConfig.dtype || "q4"
    const device = aiConfig.device || "cpu"
    const timeoutMs = aiConfig.timeoutMs || MODEL_LOAD_TIMEOUT_MS

    if (debug) {
      console.log(
        `[vibeguard] Loading AI model: ${model} (dtype=${dtype}, device=${device}, timeout=${timeoutMs}ms)`
      )
    }

    try {
      const loadPromise = transformers.pipeline("token-classification", model, {
        dtype,
        device,
      })

      // Race the model load against a timeout
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(
          () => reject(new Error(`Model load timed out after ${timeoutMs}ms`)),
          timeoutMs
        )
      })

      _pipeline = await Promise.race([loadPromise, timeoutPromise])

      if (debug) {
        console.log("[vibeguard] AI model loaded successfully")
      }
      _pipelineFailedAt = 0
      return _pipeline
    } catch (err) {
      if (debug || !aiConfig.silentFallback) {
        console.log(`[vibeguard] Failed to load AI model: ${err.message}`)
      }
      _pipeline = null
      _pipelineFailedAt = Date.now()
      return null
    }
  })()

  const result = await _loading
  _loading = null
  return result
}

/**
 * Detect PII/secrets in text using the Privacy Filter model.
 *
 * @param {string} text - Input text to scan
 * @param {object} aiConfig - AI configuration from vibeguard config
 * @param {boolean} debug - Enable debug logging
 * @returns {Promise<Array<{ start: number, end: number, original: string, category: string }>>}
 */
export async function detectWithAI(text, aiConfig, debug) {
  if (!text || typeof text !== "string" || text.length === 0) return []

  const pipe = await getPipeline(aiConfig, debug)
  if (!pipe) return []

  try {
    // Run token classification with entity aggregation
    const entities = await pipe(text, { aggregation_strategy: "simple" })
    if (!Array.isArray(entities) || entities.length === 0) return []

    const allowedCategories =
      Array.isArray(aiConfig.categories) && aiConfig.categories.length > 0
        ? new Set(aiConfig.categories.map((c) => c.toLowerCase()))
        : null

    const spans = []

    for (const entity of entities) {
      if (!entity || typeof entity !== "object") continue

      // entity_group is the label without B-/I- prefix (from aggregation)
      const rawLabel = String(entity.entity_group ?? entity.entity ?? "").toLowerCase()
      // Strip B-/I- prefix if aggregation didn't remove it
      const label = rawLabel.replace(/^[bi]-/, "")

      if (!label || label === "o") continue
      if (allowedCategories && !allowedCategories.has(label)) continue

      const start = Number(entity.start)
      const end = Number(entity.end)
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue
      if (start < 0 || end <= start || end > text.length) continue

      const original = text.slice(start, end)
      const category = LABEL_TO_CATEGORY[label] ?? label.toUpperCase()

      spans.push({ start, end, original, category })
    }

    if (debug && spans.length > 0) {
      console.log(
        `[vibeguard] AI detected ${spans.length} span(s): ${spans.map((s) => s.category).join(", ")}`
      )
    }

    return spans
  } catch (err) {
    if (!aiConfig.silentFallback || debug) {
      console.log(`[vibeguard] AI inference error: ${err.message}, falling back to regex-only`)
    }
    return []
  }
}

/**
 * Dispose the loaded model pipeline to free memory.
 */
export async function disposeAI() {
  if (_pipeline) {
    try {
      if (typeof _pipeline.dispose === "function") await _pipeline.dispose()
    } catch {
      /* ignore */
    }
    _pipeline = null
  }
}

/**
 * Check if AI detection is available (transformers package installed).
 */
export async function isAIAvailable() {
  const transformers = await loadTransformers()
  return transformers !== null
}
