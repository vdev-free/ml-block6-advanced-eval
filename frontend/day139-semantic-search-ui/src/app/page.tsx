"use client";

import { useState } from "react";

type SearchResult = {
  text: string;
  score: number;
};

type SearchResponse = {
  query: string;
  top_k: number;
  latency_ms: number;
  results: SearchResult[];
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [topK, setTopK] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async () => {
    if (!query.trim()) return;

    try {
      setLoading(true);
      setError("");
      setResults([]);
      setLatencyMs(null);
      setHasSearched(false);
      setTopK(null);

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/search`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query,
            top_k: 3,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Search request failed");
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
      setLatencyMs(data.latency_ms);
      setHasSearched(true);
      setTopK(data.top_k);
    } catch (err) {
      setError("Не вдалося отримати результати пошуку");
      setResults([]);
      setHasSearched(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-white px-6 py-10 text-black">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-bold">Semantic Search Demo</h1>
          <p className="text-sm text-gray-600">
            Введи запит і ми знайдемо найближчі тексти за змістом.
          </p>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Наприклад: I need help from customer support"
            className="w-full rounded-xl border border-gray-300 px-4 py-3 outline-none transition focus:border-black"
          />

          <button
            type="button"
            onClick={handleSearch}
            disabled={loading}
            className="rounded-xl bg-black px-5 py-3 text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        <div className="rounded-xl border border-gray-200 p-4">
          <p className="text-sm text-gray-500">Current query:</p>
          {loading && (
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm">
              Шукаю результати...
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}
          <p className="mt-2 text-base">{query || "Поки що пусто"}</p>
          {latencyMs !== null && (
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm text-gray-700">
              Search latency:{" "}
              <span className="font-semibold">{latencyMs} ms</span>
            </div>
          )}
          {hasSearched && !loading && !error && (
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
                <p className="text-xs uppercase tracking-wide text-gray-500">
                  Top K
                </p>
                <p className="mt-2 text-lg font-semibold">{topK ?? "-"}</p>
              </div>

              <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
                <p className="text-xs uppercase tracking-wide text-gray-500">
                  Results count
                </p>
                <p className="mt-2 text-lg font-semibold">{results.length}</p>
              </div>
            </div>
          )}
          {hasSearched && !loading && !error && results.length === 0 && (
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm text-gray-600">
              Нічого не знайдено за цим запитом.
            </div>
          )}
          {results.length > 0 && (
            <div className="flex flex-col gap-3">
              <h2 className="text-xl font-semibold">Results</h2>

              {results.map((result, index) => (
                <div
                  key={`${result.text}-${index}`}
                  className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm"
                >
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs uppercase tracking-wide text-gray-500">
                      Result #{index + 1}
                    </p>
                    <p className="rounded-full bg-gray-100 px-3 py-1 text-sm font-medium text-gray-700">
                      Score: {result.score.toFixed(4)}
                    </p>
                  </div>

                  <p className="mt-4 text-base leading-7 text-black">
                    {result.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
