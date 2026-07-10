# A/B report: system prompts

## Versions

- A: short support prompt focused on concise Russian answers.
- B: structured support prompt focused on safe, detailed answers and clarifying questions.

## Traffic Split

- A: 50%
- B: 50%

The split is deterministic: `sha256(owner_external_id) % 100`, so the same owner always receives the same active prompt version.

## Test Volume

This repository contains a minimal educational implementation. The current sample is synthetic and small, so it is not statistically significant.

## Feedback Ratio

The admin stats endpoint reports `feedback_up_ratio` for the last 24 hours. With the bundled JSON storage and no real users, the expected value is `null` until feedback events are collected.

## Conclusion

Keep both prompts active for the homework demo. Use real Telegram feedback over a larger sample before choosing a winner.
