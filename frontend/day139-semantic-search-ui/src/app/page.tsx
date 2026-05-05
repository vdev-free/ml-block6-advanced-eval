"use client";

import { FormEvent, useMemo, useState } from "react";

type MessageCategory =
  | "delivery_issue"
  | "general_question"
  | "payment_issue"
  | "positive_feedback"
  | "product_quality"
  | "return_refund";

type ClassifyMessageResponse = {
  category: MessageCategory;
  score: number;
  is_confident: boolean;
};

type ExampleMessage = {
  label: string;
  category: MessageCategory;
  text: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const CATEGORY_LABELS: Record<MessageCategory, string> = {
  delivery_issue: "Delivery issue",
  general_question: "General question",
  payment_issue: "Payment issue",
  positive_feedback: "Positive feedback",
  product_quality: "Product quality",
  return_refund: "Return / refund",
};

const CATEGORY_DESCRIPTIONS: Record<MessageCategory, string> = {
  delivery_issue:
    "The customer message is related to shipping, late delivery, damaged package, tracking, or pickup point.",
  general_question:
    "The customer asks a general question about availability, size, shipping, product details, or store policy.",
  payment_issue:
    "The customer has a problem with payment, checkout, card charge, invoice, or promo code.",
  positive_feedback:
    "The customer is happy with the product, delivery, support, or shopping experience.",
  product_quality:
    "The customer reports a quality problem: broken item, wrong size, scratches, poor material, or mismatch with description.",
  return_refund:
    "The customer wants to return, exchange, cancel, or get money back.",
};

const EXAMPLE_MESSAGES: ExampleMessage[] = [
  {
    label: "Late delivery",
    category: "delivery_issue",
    text: "My order arrived late and the package was damaged.",
  },
  {
    label: "Payment problem",
    category: "payment_issue",
    text: "My card was charged twice during checkout.",
  },
  {
    label: "Product quality",
    category: "product_quality",
    text: "The headphones stopped working after two days.",
  },
  {
    label: "Return request",
    category: "return_refund",
    text: "I want to return this product and get a refund.",
  },
  {
    label: "Positive feedback",
    category: "positive_feedback",
    text: "Fast delivery and great customer service, thank you!",
  },
  {
    label: "General question",
    category: "general_question",
    text: "Do you ship this product to Finland?",
  },
];

function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

export default function Home() {
  const [message, setMessage] = useState<string>(
    "My order arrived late and the package was damaged."
  );
  const [result, setResult] = useState<ClassifyMessageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const trimmedMessage = message.trim();

  const selectedCategoryDescription = useMemo(() => {
    if (!result) {
      return null;
    }

    return CATEGORY_DESCRIPTIONS[result.category];
  }, [result]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!trimmedMessage) {
      setError("Please enter a customer message.");
      setResult(null);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      setResult(null);

      const response = await fetch(`${API_BASE_URL}/classify-message`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: trimmedMessage,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = (await response.json()) as ClassifyMessageResponse;

      setResult(data);
    } catch (unknownError) {
      const message =
        unknownError instanceof Error
          ? unknownError.message
          : "Something went wrong. Please try again.";

      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleExampleClick(example: ExampleMessage) {
    setMessage(example.text);
    setResult(null);
    setError(null);
  }

  return (
    <main className="page">
      <section className="hero">
        <div className="eyebrow">Mini Production Project · Day 149</div>

        <h1>AI E-commerce Message Classifier</h1>

        <p className="heroText">
          Paste a customer message from an online store and classify it into a
          support category: delivery, payment, product quality, return/refund,
          positive feedback, or general question.
        </p>
      </section>

      <section className="layout">
        <form className="card inputCard" onSubmit={handleSubmit}>
          <div className="sectionHeader">
            <div>
              <h2>Customer message</h2>
              <p>
                This is what a support/admin user could paste from an e-commerce
                inbox.
              </p>
            </div>
          </div>

          <label className="label" htmlFor="message">
            Message
          </label>

          <textarea
            id="message"
            className="textarea"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Example: My package arrived damaged and two days late."
            rows={8}
          />

          <div className="actions">
            <button
              className="primaryButton"
              type="submit"
              disabled={isLoading || !trimmedMessage}
            >
              {isLoading ? "Analyzing..." : "Analyze message"}
            </button>

            <span className="hint">
              API: <code>{API_BASE_URL}</code>
            </span>
          </div>

          {error && <div className="errorBox">{error}</div>}
        </form>

        <aside className="card resultCard">
          <div className="sectionHeader">
            <div>
              <h2>AI result</h2>
              <p>Prediction from the FastAPI transformer service.</p>
            </div>
          </div>

          {!result && !isLoading && (
            <div className="emptyState">
              <div className="emptyIcon">🤖</div>
              <p>Run analysis to see the predicted support category.</p>
            </div>
          )}

          {isLoading && (
            <div className="emptyState">
              <div className="emptyIcon">⏳</div>
              <p>Model is analyzing the message...</p>
            </div>
          )}

          {result && (
            <div className="prediction">
              <div className="predictionTop">
                <span className="categoryBadge">
                  {CATEGORY_LABELS[result.category]}
                </span>

                <span
                  className={
                    result.is_confident
                      ? "confidenceBadge confidenceGood"
                      : "confidenceBadge confidenceLow"
                  }
                >
                  {result.is_confident ? "Confident" : "Low confidence"}
                </span>
              </div>

              <div className="scoreBlock">
                <span className="scoreLabel">Confidence score</span>
                <strong>{formatScore(result.score)}</strong>
              </div>

              <div className="scoreBar">
                <div
                  className="scoreBarFill"
                  style={{ width: `${Math.round(result.score * 100)}%` }}
                />
              </div>

              {selectedCategoryDescription && (
                <p className="description">{selectedCategoryDescription}</p>
              )}

              {!result.is_confident && (
                <div className="warningBox">
                  Low confidence — please review manually. The model was trained
                  on a very small educational dataset and is not
                  production-ready.
                </div>
              )}
            </div>
          )}
        </aside>
      </section>

      <section className="card examplesCard">
        <div className="sectionHeader">
          <div>
            <h2>Example messages</h2>
            <p>Click an example to test the classifier quickly.</p>
          </div>
        </div>

        <div className="examplesGrid">
          {EXAMPLE_MESSAGES.map((example) => (
            <button
              key={example.label}
              className="exampleButton"
              type="button"
              onClick={() => handleExampleClick(example)}
            >
              <span>{example.label}</span>
              <small>{CATEGORY_LABELS[example.category]}</small>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
