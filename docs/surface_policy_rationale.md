# Surface-policy condition

The primary bias-fusion experiment must measure an ordinary next-token policy, not agreement on experiment-only trace-control classes.

The main `surface_v3` condition therefore has these model-facing properties:

- equality transitions use the literal `=` vocabulary token;
- generation terminates with the normal `<EOS>` token;
- `<EQ_STEP>` and `<TRACE_STOP>` are generator-facing aliases only and are not vocabulary entries;
- `+`, `,`, `[`, and `]` are literal surface tokens;
- the trained common base learns identity equality (`x = x`) and EOS syntax without learning operator answers.

The typed-token v2 condition is retained only as a diagnostic ablation. A result that appears only in the typed condition is not sufficient evidence that bias fusion transfers to ordinary continuation policies.

This remains a controlled synthetic proxy. Atomic integer tokens and prompt-only operator/task markers are deliberate simplifications. A later digit- or subword-tokenized replication is required before generalizing the result to unrestricted natural-language models.
