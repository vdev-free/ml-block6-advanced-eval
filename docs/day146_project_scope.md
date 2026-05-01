# Day 146 — AI E-commerce Message Classifier: Project Scope

## Project name

AI E-commerce Message Classifier

## Goal

Build a small production-style AI feature for an e-commerce product.

The app will classify customer messages into business categories such as delivery problems, payment issues, product quality feedback, return/refund requests, positive feedback, and general questions.

## Business scenario

An online store receives many customer messages every day:

- product reviews
- delivery complaints
- payment questions
- return/refund requests
- general support questions

Instead of reading every message manually from the start, a support/admin user can use AI to quickly understand what the message is about and how urgent it may be.

## Message categories

The model will classify each customer message into one of these categories:

| Category | Meaning | Example |
|---|---|---|
| `delivery_issue` | Problem with shipping, late delivery, damaged package, missing parcel | "My order arrived late and the box was damaged." |
| `payment_issue` | Problem with payment, checkout, card, invoice, refund payment | "My card was charged twice during checkout." |
| `product_quality` | Product is broken, low quality, wrong size, not as described | "The headphones stopped working after two days." |
| `return_refund` | Customer wants to return an item or get a refund | "I want to return this product and get my money back." |
| `positive_feedback` | Customer is happy with product or service | "Great quality and fast delivery, thank you!" |
| `general_question` | General question before/after purchase | "Do you ship this product to Finland?" |

## API contract

### Endpoint

`POST /classify-message`

### Request

```json
{
  "message": "My order arrived late and the package was damaged."
}

## UI scope

The UI will be a small Next.js page for an e-commerce admin/support user.

### Main UI elements

- Textarea for customer message
- Analyze button
- Result card with predicted category
- Confidence score
- Low-confidence warning
- Example messages for quick testing

### Example UI flow

1. Admin pastes a customer message.
2. Admin clicks "Analyze".
3. UI sends request to FastAPI backend.
4. Backend returns category, score, and confidence flag.
5. UI displays the result as a clear support badge.

### Example result

`delivery_issue` → show badge: "Delivery issue"

If `is_confident` is false, show:

"Low confidence — please review manually."

## 5-day implementation plan

### Day 146 — Scope

Define project goal, business use case, categories, API contract, UI scope, and definition of done.

### Day 147 — Training

Create a small e-commerce message dataset and fine-tune a transformer classifier for message categories.

### Day 148 — API

Build a FastAPI backend with:

- `GET /health`
- `GET /model-info`
- `POST /classify-message`

The API will return category, score, and confidence flag.

### Day 149 — UI

Build a small Next.js UI with:

- customer message textarea
- example messages
- analyze button
- result card
- confidence warning

### Day 150 — Polish + demo

Add README, demo screenshot, curl examples, known limitations, and improvement plan.

## Definition of done

The mini production project is done when the repository contains:

- A documented project scope
- A small labeled e-commerce message dataset
- A trained transformer classifier
- Evaluation results and known limitations
- FastAPI backend with validation and confidence threshold
- Next.js UI for testing customer messages
- Demo screenshot
- README with setup instructions, API examples, and improvement plan

## Known limitations

This is a small educational production-style project.

Expected limitations:

- Dataset will be small and synthetic/semi-synthetic
- Model may confuse similar categories such as `delivery_issue` and `return_refund`
- Confidence score must be shown to avoid over-trusting weak predictions
- Real production use would require more real customer messages, privacy review, monitoring, and continuous evaluation