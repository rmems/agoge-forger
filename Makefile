# Thin alias over the justfile (#205). The recipes live in `justfile`, so this list
# cannot drift from it; install `just` (see README "Using just") to use these targets.
TARGETS := setup check check-torch train-smoke eval-smoke test lint

.PHONY: $(TARGETS)

$(TARGETS):
	just $@
