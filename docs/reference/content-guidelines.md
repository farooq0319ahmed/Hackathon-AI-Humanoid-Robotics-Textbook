---
sidebar_position: 4
---

# Content Guidelines

This document outlines the standards and best practices for creating content in the Humanoid Robotics Book.

## Technical Accuracy Standards

### Verification Requirements
- All code examples must be tested and verified as functional
- Commands must be reproducible with exact expected outcomes
- Claims about technology, APIs, or frameworks must be fact-checked against current documentation
- All content must be grounded in verified, official documentation from authoritative sources

### Documentation Sources
- Prioritize official ROS 2, Gazebo, Unity, and NVIDIA Isaac documentation
- Use current, up-to-date resources (check publication/revision dates)
- Cite sources appropriately with links to official documentation
- Avoid third-party tutorials or unofficial sources as primary references

## Reproducibility Standards

### Command Verification
- Every command, example, and workflow must be tested on target hardware platforms
- Include expected outputs for each code example
- Document specific environment requirements (OS, versions, dependencies)
- Test all examples on both workstation and Jetson platforms where applicable

### Example Structure
- Start with clear prerequisites
- Provide step-by-step instructions
- Include expected outputs or results
- Add troubleshooting tips for common issues
- Verify 95% success rate on target hardware

## Writing Standards

### Developer-Friendly Writing
- Use precise, actionable language
- Include real examples with expected outcomes
- Make instructions reproduceable with exact commands
- Write for accessibility but maintain technical accuracy

### Consistent Structure
- Follow standardized chapter structure: objectives, examples, and summary sections
- Use consistent terminology and naming conventions
- Apply uniform formatting following Docusaurus-compatible Markdown standards
- Include learning objectives at the beginning of each section

## Formatting Guidelines

### Markdown Standards
- Use Docusaurus-compatible Markdown formatting
- Apply proper heading hierarchy (h1 for page titles, h2 for sections, etc.)
- Format code blocks with appropriate language identifiers
- Use proper list formatting (ordered for steps, unordered for options)

### Code Examples
- Include language identifiers in code blocks (```python, ```bash, etc.)
- Use realistic, practical examples rather than abstract concepts
- Include comments explaining complex code sections
- Follow language-specific style guides (PEP 8 for Python, etc.)

## Quality Assurance

### Review Process
- All content must undergo technical review by domain experts
- Hands-on exercises must be tested by target audience
- Verify all commands work on target hardware (workstation and Jetson)
- Confirm each module contains 1,500-2,500 words of comprehensive content

### Validation Checklist
- [ ] All code examples tested and verified
- [ ] Commands reproduce with 95% success rate
- [ ] Content formatted in Docusaurus-compatible Markdown
- [ ] Docusaurus builds without errors or warnings
- [ ] Technical accuracy verified against official documentation
- [ ] Consistent terminology and formatting applied
- [ ] Proper attribution to official documentation sources