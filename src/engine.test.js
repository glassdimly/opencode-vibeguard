import { describe, it } from "node:test"
import assert from "node:assert/strict"
import { redactText, redactTextWithAI } from "./engine.js"
import { buildPatternSet } from "./patterns.js"
import { PlaceholderSession } from "./session.js"
import { restoreText } from "./restore.js"

describe("redactText (regex-only)", () => {
  it("redacts email addresses", () => {
    const patterns = buildPatternSet({ builtin: ["email"], exclude: ["example.com"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Contact me at alice@corp.io for details", patterns, session)
    assert.ok(!result.text.includes("alice@corp.io"), "email should be redacted")
    assert.ok(result.text.includes("__VG_EMAIL_"), "should contain placeholder")
    assert.equal(result.matches.length, 1)
    assert.equal(result.matches[0].category, "EMAIL")
  })

  it("redacts keyword values", () => {
    const patterns = buildPatternSet({
      keywords: [{ value: "sk-abc123secret", category: "API_KEY" }],
    })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Key is sk-abc123secret here", patterns, session)
    assert.ok(!result.text.includes("sk-abc123secret"))
    assert.ok(result.text.includes("__VG_API_KEY_"))
  })

  it("respects exclude list", () => {
    const patterns = buildPatternSet({
      builtin: ["email"],
      exclude: ["test@example.com"],
    })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Email: test@example.com", patterns, session)
    assert.ok(result.text.includes("test@example.com"), "excluded email should remain")
  })

  it("handles overlapping spans correctly", () => {
    const patterns = buildPatternSet({
      keywords: [
        { value: "secret-key-abc", category: "KEY" },
        { value: "key-abc", category: "KEY_PART" },
      ],
    })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Found secret-key-abc here", patterns, session)
    assert.ok(!result.text.includes("secret-key-abc"))
    assert.ok(!result.text.includes("key-abc"))
  })

  it("returns unchanged text when nothing matches", () => {
    const patterns = buildPatternSet({ builtin: ["email"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Nothing sensitive here", patterns, session)
    assert.equal(result.text, "Nothing sensitive here")
    assert.equal(result.matches.length, 0)
  })
})

describe("new builtin patterns", () => {
  it("redacts US phone numbers", () => {
    const patterns = buildPatternSet({ builtin: ["phone_us"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Call me at (555) 123-4567 please", patterns, session)
    assert.ok(!result.text.includes("(555) 123-4567"))
    assert.ok(result.text.includes("__VG_PHONE_US_"))
  })

  it("redacts SSNs", () => {
    const patterns = buildPatternSet({ builtin: ["ssn"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("SSN: 123-45-6789", patterns, session)
    assert.ok(!result.text.includes("123-45-6789"))
    assert.ok(result.text.includes("__VG_SSN_"))
  })

  it("redacts credit card numbers", () => {
    const patterns = buildPatternSet({ builtin: ["credit_card"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Card: 4111-1111-1111-1111", patterns, session)
    assert.ok(!result.text.includes("4111-1111-1111-1111"))
    assert.ok(result.text.includes("__VG_CREDIT_CARD_"))
  })

  it("redacts private key headers", () => {
    const patterns = buildPatternSet({ builtin: ["private_key_header"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("-----BEGIN RSA PRIVATE KEY-----\nfoo", patterns, session)
    assert.ok(!result.text.includes("-----BEGIN RSA PRIVATE KEY-----"))
    assert.ok(result.text.includes("__VG_PRIVATE_KEY_"))
  })

  it("redacts AWS access keys", () => {
    const patterns = buildPatternSet({ builtin: ["aws_access_key"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("key: AKIAIOSFODNN7EXAMPLE", patterns, session)
    assert.ok(!result.text.includes("AKIAIOSFODNN7EXAMPLE"))
    assert.ok(result.text.includes("__VG_AWS_ACCESS_KEY_"))
  })

  it("redacts bearer tokens", () => {
    const patterns = buildPatternSet({ builtin: ["generic_bearer"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const result = redactText("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.xyz", patterns, session)
    assert.ok(!result.text.includes("eyJhbGciOiJIUzI1NiJ9"))
    assert.ok(result.text.includes("__VG_BEARER_TOKEN_"))
  })
})

describe("redactTextWithAI (graceful fallback)", () => {
  it("falls back to regex-only when AI is unavailable", async () => {
    const patterns = buildPatternSet({ builtin: ["email"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })

    // AI enabled but transformers not installed in test env = silent fallback
    const aiConfig = {
      enabled: true,
      model: "openai/privacy-filter",
      dtype: "q4",
      device: "cpu",
      categories: [],
      silentFallback: true,
    }

    const result = await redactTextWithAI(
      "Email alice@corp.io from Alice Smith",
      patterns,
      session,
      aiConfig,
      false
    )

    // Email should be redacted by regex even when AI is unavailable
    assert.ok(!result.text.includes("alice@corp.io"))
    assert.ok(result.text.includes("__VG_EMAIL_"))
    // "Alice Smith" won't be redacted without AI, which is expected fallback
    assert.ok(result.text.includes("Alice Smith"))
  })

  it("handles empty text", async () => {
    const patterns = buildPatternSet({ builtin: ["email"] })
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const aiConfig = { enabled: true, model: "openai/privacy-filter", dtype: "q4", device: "cpu", categories: [], silentFallback: true }

    const result = await redactTextWithAI("", patterns, session, aiConfig, false)
    assert.equal(result.text, "")
    assert.equal(result.matches.length, 0)
  })
})

describe("PlaceholderSession", () => {
  it("produces stable placeholders for same input", () => {
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const ph1 = session.getOrCreatePlaceholder("secret123", "API_KEY")
    const ph2 = session.getOrCreatePlaceholder("secret123", "API_KEY")
    assert.equal(ph1, ph2)
  })

  it("supports bidirectional lookup", () => {
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const ph = session.getOrCreatePlaceholder("myvalue", "TEXT")
    assert.equal(session.lookup(ph), "myvalue")
    assert.equal(session.lookupReverse("myvalue"), ph)
  })
})

describe("restoreText", () => {
  it("restores placeholders to original values", () => {
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const ph = session.getOrCreatePlaceholder("alice@corp.io", "EMAIL")
    const restored = restoreText(`Contact ${ph} for details`, session)
    assert.equal(restored, "Contact alice@corp.io for details")
  })

  it("leaves unknown placeholders unchanged", () => {
    const session = new PlaceholderSession({ prefix: "__VG_" })
    const text = "Some __VG_UNKNOWN_abcdef012345__ here"
    const restored = restoreText(text, session)
    assert.equal(restored, text)
  })
})

describe("config normalizeAiConfig", async () => {
  // Import config module to test normalization
  const { loadConfig } = await import("./config.js")

  it("defaults ai to disabled", async () => {
    // loadConfig with a non-existent dir returns enabled=false
    const cfg = await loadConfig("/nonexistent-dir-for-test-" + Date.now())
    assert.equal(cfg.ai.enabled, false)
    assert.equal(cfg.ai.model, "openai/privacy-filter")
    assert.equal(cfg.ai.dtype, "q4")
    assert.equal(cfg.ai.device, "cpu")
    assert.equal(cfg.ai.silentFallback, true)
    assert.deepEqual(cfg.ai.categories, [])
  })
})
