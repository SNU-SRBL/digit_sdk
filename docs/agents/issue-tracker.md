# Issue tracker: Local Markdown

Issues and PRDs for this repository live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is a `Status:` line near the top of each issue file
- Comments and conversation history append under a `## Comments` heading

When a workflow says to publish an issue, create the corresponding local file. When it says to fetch a ticket, read the referenced local file. Do not create or update GitHub issues.
