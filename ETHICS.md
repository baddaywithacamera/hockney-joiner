# Ethics & AI Disclosure

**Hockney Joiner Assembly Tool** · GPL v3 · Sean McCormick

---

## AI Assistance in Development

This software was developed with AI assistance (Claude, by Anthropic).
AI was used for: code drafting, architecture discussion, documentation,
and debugging. All design decisions, editorial choices, and final code
review were made by the human author.

This disclosure follows the same standard as SNAPSMACK.

## AI in the Application

The application uses **LightGlue** (open source, ETH Zurich / CVG group)
for keypoint matching between photographs. LightGlue does one thing:
detect overlapping regions between image pairs and suggest placement.

It does not generate image content. It does not hallucinate.
It does not send data anywhere — the model runs fully offline after
the one-time download.

## What This Tool Does Not Do

- It does not modify your original photographs. Ever.
- It does not send your images to any server.
- It does not require an internet connection after first install.
- It does not make creative decisions on your behalf.
  The photographer decides what stays, what goes, and what's crooked.

## License

GPL v3. Full text in LICENSE.
Source code: https://github.com/[your-username]/hockney-joiner
