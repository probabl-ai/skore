---
parent: Decisions
nav_order: 100
title: Don't use Pydantic apart from FastAPI

# These are optional elements. Feel free to remove any of them.
# status: {proposed | rejected | accepted | deprecated | … | superseded by [ADR-0005](0005-example.md)}
# date: {YYYY-MM-DD when the decision was last updated}
# deciders: {list everyone involved in the decision}
# consulted: {list everyone whose opinions are sought (typically subject-matter experts); and with whom there is a two-way communication}
# informed: {list everyone who is kept up-to-date on progress; and with whom there is a one-way communication}
---
<!-- we need to disable MD025, because we use the different heading "ADR Template" in the homepage (see above) than it is foreseen in the template -->
<!-- markdownlint-disable-next-line MD025 -->
# Don't use Pydantic apart from FastAPI

## Context and Problem Statement

In the Mandr project, we save free-form data into a Store (e.g. a [DiskCache](https://grantjenks.com/docs/diskcache/) database) and then serialize it to send to our frontend.

Here are the reasons we can think of that brought us to consider Pydantic; namely, in one line, it can:
- Perform serialization of a Mandr, which we need to do when sending data to the frontend
- Validate the data inserted into a Mandr, which we might want to do
- Export a data transfer specification (e.g. JSONSchema), rather than writing one from scratch, which we have been doing so far

Here are the reasons why we have determined Pydantic to be ill-adapted for us right now:
- Pydantic is typically used for validating data coming from the outside, and less so for serializing data. However, we use it more for serialization than validation
- It's overkill: we don't need this much control on the different kinds of data inserted into a Mandr
- It forces us to determine in advance how to serialize every data type, which we might not want to do (leaving that to the user, while defining sensible defaults)
- It contaminates the codebase; it's hard to transition to and from Pydantic-free code
- It is unwieldy for our purposes: our main model consists of a union of `BaseModel`, which itself is not a `BaseModel`. As such, it requires some "magic" to be used (see <https://typethepipe.com/post/pydantic-discriminated-union/> or <https://blog.det.life/pydantic-for-experts-discriminated-unions-in-pydantic-v2-2d9ca965b22f>) which makes the code harder to approach
- The import time is likely dominated by `import pydantic`, although this remains to be tested (see <https://github.com/probabl-ai/mandr/issues/187>)
- Pydantic serialization doesn't allow us to easily deal with all use-cases (e.g. numpy arrays, see <https://github.com/probabl-ai/mandr/issues/197>)

<!-- {Describe the context and problem statement, e.g., in free form using two to three sentences or in the form of an illustrative story.
 You may want to articulate the problem in form of a question and add links to collaboration boards or issue management systems.} -->

<!-- This is an optional element. Feel free to remove. -->
## Decision Drivers

* {decision driver 1, e.g., a force, facing concern, …}
* {decision driver 2, e.g., a force, facing concern, …}
* … <!-- numbers of drivers can vary -->

## Considered Options

* {title of option 1}
* {title of option 2}
* {title of option 3}
* … <!-- numbers of options can vary -->

## Decision Outcome

Chosen option: "{title of option 1}", because
{justification. e.g., only option, which meets k.o. criterion decision driver | which resolves force {force} | … | comes out best (see below)}.

<!-- This is an optional element. Feel free to remove. -->
### Consequences

* Good, because {positive consequence, e.g., improvement of one or more desired qualities, …}
* Bad, because {negative consequence, e.g., compromising one or more desired qualities, …}
* … <!-- numbers of consequences can vary -->

<!-- This is an optional element. Feel free to remove. -->
## Validation

{describe how the implementation of/compliance with the ADR is validated. E.g., by a review or an ArchUnit test}

<!-- This is an optional element. Feel free to remove. -->
## Pros and Cons of the Options

### {title of option 1}

<!-- This is an optional element. Feel free to remove. -->
{example | description | pointer to more information | …}

* Good, because {argument a}
* Good, because {argument b}
<!-- use "neutral" if the given argument weights neither for good nor bad -->
* Neutral, because {argument c}
* Bad, because {argument d}
* … <!-- numbers of pros and cons can vary -->

### {title of other option}

{example | description | pointer to more information | …}

* Good, because {argument a}
* Good, because {argument b}
* Neutral, because {argument c}
* Bad, because {argument d}
* …

<!-- This is an optional element. Feel free to remove. -->
## More Information

{You might want to provide additional evidence/confidence for the decision outcome here and/or
 document the team agreement on the decision and/or
 define when this decision when and how the decision should be realized and if/when it should be re-visited and/or
 how the decision is validated.
 Links to other decisions and resources might here appear as well.}
