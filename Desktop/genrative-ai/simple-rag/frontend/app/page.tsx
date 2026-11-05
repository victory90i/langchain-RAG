"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Source = { page: number | null; text: string };

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  cached?: string | null;
  error?: string; // shown in red under the answer; any partial answer is kept
};

type StreamEvent =
  | { type: "meta"; cached: boolean; how: string | null; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

type Health = { state: "checking" } | { state: "ok"; chunks: number } | { state: "down" };

// ---------- helpers ----------
async function pingHealth(): Promise<Health> {
  try {
    const res = await fetch(`${API}/api/health`, { cache: "no-store" });
    if (!res.ok) return { state: "down" };
    const data = await res.json();
    return { state: "ok", chunks: Number(data.chunks) || 0 };
  } catch {
    return { state: "down" };
  }
}

// Turns the response body of a failed HTTP call (FastAPI sends {"detail": ...}) into text
async function readHttpError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail)) {
      const msgs = data.detail
        .map((d: { msg?: string }) => d?.msg)
        .filter(Boolean)
        .join("; ");
      if (msgs) return msgs;
    }
  } catch {
    // body was not JSON; fall through to the generic message
  }
  return `Server error (${res.status}).`;
}

function describeError(e: unknown): string {
  if (e instanceof DOMException && e.name === "AbortError") return "Stopped.";
  if (e instanceof TypeError) {
    // fetch() throws a TypeError when the request never reaches the server:
    // backend not running, wrong URL, or the browser blocked it (CORS).
    return (
      `Can't reach the API at ${API}. Make sure the backend is running ` +
      `(uvicorn main:app --port 8000) and that CORS_ORIGINS includes ${window.location.origin}.`
    );
  }
  return e instanceof Error ? e.message : "Something went wrong.";
}

// ---------- page ----------
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [useCache, setUseCache] = useState(true);
  const [health, setHealth] = useState<Health>({ state: "checking" });
  const [notice, setNotice] = useState("");

  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Check the backend once on load
  useEffect(() => {
    let cancelled = false;
    pingHealth().then((h) => {
      if (!cancelled) setHealth(h);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Cancel any in-flight request when leaving the page
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Update only the last (assistant) message while it streams in
  const patchLast = (fn: (m: Message) => Message) =>
    setMessages((prev) => prev.map((m, i) => (i === prev.length - 1 ? fn(m) : m)));

  function retryHealth() {
    setHealth({ state: "checking" });
    pingHealth().then(setHealth);
  }

  async function send() {
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    let receivedText = false;
    let finished = false;
    let failed = false;

    const handleEvent = (ev: StreamEvent) => {
      switch (ev.type) {
        case "meta":
          patchLast((m) => ({
            ...m,
            sources: ev.sources,
            cached: ev.cached ? ev.how : null,
          }));
          break;
        case "token":
          receivedText = true;
          patchLast((m) => ({ ...m, content: m.content + ev.text }));
          break;
        case "done":
          finished = true;
          break;
        case "error":
          failed = true;
          patchLast((m) => ({ ...m, error: ev.message }));
          break;
      }
    };

    // One NDJSON line -> one event. A bad line is skipped instead of crashing the chat.
    const handleLine = (line: string) => {
      if (!line.trim()) return;
      try {
        handleEvent(JSON.parse(line) as StreamEvent);
      } catch {
        /* ignore malformed line */
      }
    };

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, use_cache: useCache }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(await readHttpError(res));
      if (!res.body) throw new Error("The server sent an empty response.");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // A network chunk can hold partial lines, so buffer until a newline
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        lines.forEach(handleLine);
      }
      handleLine(buffer); // last line, if the stream didn't end with a newline

      if (!finished && !failed) {
        patchLast((m) => ({
          ...m,
          error: receivedText ? "The response was cut off." : "No response received.",
        }));
      }
      setHealth((h) => (h.state === "down" ? { state: "checking" } : h));
    } catch (e) {
      patchLast((m) => ({ ...m, error: describeError(e) }));
      if (e instanceof TypeError) setHealth({ state: "down" });
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function clearCache() {
    try {
      const res = await fetch(`${API}/api/cache`, { method: "DELETE" });
      if (!res.ok) throw new Error();
      setNotice("Cache cleared.");
    } catch {
      setNotice("Couldn't clear the cache (is the backend running?).");
    }
    setTimeout(() => setNotice(""), 3000);
  }

  return (
    <main className="mx-auto flex h-screen max-w-3xl flex-col px-4 py-6">
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-semibold">📄 Chat with your document</h1>
        <div className="flex items-center gap-3 text-sm">
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={useCache}
              onChange={(e) => setUseCache(e.target.checked)}
            />
            Use cache
          </label>
          <button
            onClick={clearCache}
            className="rounded border px-2 py-1 hover:bg-black/5 dark:hover:bg-white/10"
          >
            Clear cache
          </button>
        </div>
      </header>

      <div className="mb-3 flex items-center gap-2 text-xs">
        {health.state === "checking" && <span className="opacity-60">Checking backend…</span>}
        {health.state === "ok" && (
          <span className="text-green-600 dark:text-green-400">
            ● Backend connected · {health.chunks} chunks indexed
          </span>
        )}
        {health.state === "down" && (
          <>
            <span className="text-red-600 dark:text-red-400">● Backend unreachable at {API}</span>
            <button onClick={retryHealth} className="rounded border px-2 py-0.5 hover:bg-black/5 dark:hover:bg-white/10">
              Retry
            </button>
          </>
        )}
        {notice && <span className="ml-auto opacity-70">{notice}</span>}
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto rounded-lg border p-4">
        {messages.length === 0 && (
          <p className="text-sm opacity-60">Ask a question about the document to get started.</p>
        )}

        {messages.map((m, i) => {
          const isLast = i === messages.length - 1;
          return (
            <div key={i} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
              <div
                className={`max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
                  m.role === "user" ? "bg-blue-600 text-white" : "bg-black/5 dark:bg-white/10"
                }`}
              >
                {m.content || (loading && isLast && !m.error ? "Thinking…" : "")}

                {m.error && (
                  <div className="mt-2 rounded bg-red-100 px-2 py-1 text-xs text-red-900 dark:bg-red-950 dark:text-red-200">
                    ⚠️ {m.error}
                  </div>
                )}

                {m.cached && <div className="mt-2 text-xs opacity-60">⚡ From cache ({m.cached})</div>}

                {m.sources && m.sources.length > 0 && (
                  <details className="mt-2 text-xs opacity-80">
                    <summary className="cursor-pointer">Retrieved context</summary>
                    {m.sources.map((s, j) => (
                      <div key={j} className="mt-2">
                        <strong>
                          [{j + 1}] {s.page !== null ? `page ${s.page + 1}` : "n/a"}
                        </strong>
                        <p>{s.text}</p>
                      </div>
                    ))}
                  </details>
                )}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="mt-4 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something about the document"
          disabled={loading}
          className="flex-1 rounded-lg border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-blue-500"
        />
        {loading ? (
          <button
            type="button"
            onClick={stop}
            className="rounded-lg border px-4 py-2 text-sm hover:bg-black/5 dark:hover:bg-white/10"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white disabled:opacity-50"
          >
            Send
          </button>
        )}
      </form>
    </main>
  );
}