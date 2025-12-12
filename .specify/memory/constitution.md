<!-- SYNC IMPACT REPORT
Version change: N/A -> 1.0.0
Modified principles: N/A
Added sections: Core Principles (6 principles added), Key Standards, Constraints, Success Criteria
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md ✅ updated
- .specify/templates/spec-template.md ✅ updated
- .specify/templates/tasks-template.md ✅ updated
- .specify/templates/commands/*.md ⚠ pending
- README.md ⚠ pending
Follow-up TODOs: None
-->

# AI/Spec-Driven Book Creation Constitution

## Core Principles

### I. Spec-driven, Modular Content Generation
Spec-driven development approach where every chapter begins with a detailed specification; Content modules must be self-contained, independently reviewable, and reusable; Clear purpose required - no placeholder or organizational-only content.

### II. Technical Accuracy Based on Official Documentation
All content must be grounded in verified, official documentation from authoritative sources; Claims about technology, APIs, or frameworks must be fact-checked against current documentation; Third-party tutorials or unofficial sources are not acceptable as primary references.

### III. Clear, Developer-Friendly Writing (NON-NEGOTIABLE)
Content must follow developer-first writing principles: precise, actionable, with real examples; Language should be accessible but technically accurate; All instructions must be reproduceable with exact commands and expected outcomes.

### IV. Consistent Structure, Terminology, and Formatting
Standardized chapter structure required: objectives, examples, and summary sections; Consistent terminology and naming conventions across all content; Uniform formatting following Docusaurus-compatible Markdown standards.

### V. Reproducible Workflows and Commands
Every command, example, and workflow must be tested and verified as functional; Real commands and examples only - no hallucinated features or hypothetical scenarios; All instructions must produce predictable, repeatable results.

### VI. Docusaurus and GitHub Pages Compatibility
All content must be compatible with Docusaurus 3+ and suitable for GitHub Pages deployment; Proper Markdown formatting, navigation structure, and asset handling required; Build process must complete without errors.

## Key Standards

Use Docusaurus-compatible Markdown formatting throughout all content.
Real commands, real examples — no hallucinated features or unverified information.
Each chapter includes objectives, examples, and summary sections.
All factual claims must be verifiable through official documentation.
Consistent folder, file, and naming conventions following Docusaurus standards.

## Constraints

Minimum 8 chapters required, each 1500–2500 words in length.
Must be compatible with Docusaurus 3+ and GitHub Pages deployment.
Content generated through Spec-Kit Plus + Claude Code workflow.
Output must build and deploy successfully without errors.
All content must be technically accurate and reproduceable.

## Success Criteria

Every chapter matches its corresponding specification document.
Docusaurus builds without errors or warnings.
GitHub Pages deployment passes all checks and deploys successfully.
All instructions are reproducible and technically correct.
Final book product is clear, consistent, and complete.

## Governance

This constitution supersedes all other development practices and guidelines; All amendments require formal documentation, approval, and migration planning. All pull requests and reviews must verify compliance with these principles; Complexity must be justified with clear benefits; Use official Docusaurus and GitHub documentation for runtime development guidance.

**Version**: 1.0.0 | **Ratified**: 2025-12-12 | **Last Amended**: 2025-12-12
