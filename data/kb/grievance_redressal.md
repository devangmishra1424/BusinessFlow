# Grievance redressal policy

A grievance is different from a routine dispute or restructuring
request: it's a borrower saying they are unhappy with HOW their case was
handled -- by this agent, or by a human on this side -- not a
disagreement about a specific amount or date. "I want to file a
complaint" and "I'm not happy with how this was handled" are grievances
even when no fee or figure is being disputed at all.

## Fair treatment is a baseline, not a favour

As a regulated NBFC, this lender operates under RBI's Fair Practices
Code, which sets out a borrower's right to fair, transparent, and
non-coercive treatment in collections -- and requires the lender to run a
grievance redressal mechanism a borrower can actually use. Acknowledging
a complaint and taking it seriously is standard practice here, not
something being granted as a special exception.

## Internal escalation, not the routine kind

A grievance still goes to a human via escalation, the same mechanism
escalation_policy.md describes for any other out-of-policy request -- but
the reason given must say plainly that this is a grievance about how the
account was handled, not a routine account matter. A human triaging a
queue needs that distinction up front to route it correctly, the same way
they need to know a fraud claim (see agent/client.py) isn't an ordinary
dispute.

## If the borrower is still unsatisfied

A borrower who says the internal response hasn't resolved things has the
right to take the complaint outside this business, to the RBI Banking/
NBFC Ombudsman -- a real, regulator-run escalation path, not something
this agent is offering as a courtesy.

There is no verified, current phone number, address, or web link for the
Ombudsman anywhere in this system, and none should ever be stated from
memory -- an invented contact detail would be exactly the kind of
fabrication this system's grounding rules exist to prevent. Tell the
borrower the path exists and that a human handling their account can give
them the actual current contact details for it, rather than guessing at
specifics that aren't grounded anywhere here.
