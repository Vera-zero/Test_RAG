"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["entity_to_timeline_analysis"]= """
You are given an entity name and a list of events related to that entity.  
Each event in the list contains:  
- sentence (a short description of the event)  
- context (additional context for the event)  
- start_time (timestamp or date indicating when the event began)  
- end_time (timestamp or date indicating when the event ended, if applicable)  

Analyze the event list and determine whether there are events that can form a **timeline** (i.e., events that are chronologically connected or thematically sequential).  

If timelines exist (one or more), output :  
- timelines: a list where each element contains:  
  - timeline_name (a concise name for the timeline)  
  - timeline_description (a brief description explaining the timeline's theme or sequence)  

If no timelines can be formed, output an empty list for timelines.  

Output Format:
{
  "timelines": [
    {
      "timeline_name": "example_name",
      "timeline_description": "example_description"
    }
  ]
}

Example :
Input:

Entity: "CompanyXYZ"  
Events:  
1. { "sentence": "CompanyXYZ founded", "context": "tech industry", "start_time": "2005-01-01", "end_time": null }  
2. { "sentence": "First product launch", "context": "product release", "start_time": "2006-03-15", "end_time": null }  
3. { "sentence": "Expanded to Europe", "context": "global expansion", "start_time": "2010-08-01", "end_time": null }  

Output:

{
  "timelines": [
    {
      "timeline_name": "CompanyXYZ Growth Milestones",
      "timeline_description": "Chronological key events from founding to European expansion."
    }
  ]
}

Now process the following input:

Entity: [entity_name]  
Events: [event_list]  

"""

PROMPTS["precise_time"] = """
You are a precise time annotation assistant. Your task is to identify fuzzy time expressions in text and convert them to absolute dates (YYYY-MM-DD format).  

Input:  
A document containing time-related information. 

Processing Steps:
1. Locate all fuzzy time expressions in the document (e.g., "18 years old", "2 years later", "last month").  
2. For each fuzzy time, determine the **reference date** (use today's date as the default reference if not specified in context) and calculate the corresponding absolute date.  
3. Output a structured list of mappings for this document in valid JSON format. Each mapping should include:  
   -"fuzzy_time": the original text of the fuzzy expression.  
   -"absolute_date": the calculated date in (YYYY,MM,DD) format.IF only extract the year message,format as (YYYY),if only extract the year and the month message,format as (YYYY,MM).
   -"context_snippet": a short surrounding text snippet for reference.  

Output Format: 
[
  {
    "fuzzy_time": "18 years old",
    "absolute_date": "(2006,12,22)",
    "context_snippet": "When he was 18 years old, he entered college."
  }
]

Now, process the following text chunk:

[Text Chunk Placeholder]
"""

PROMPTS["dynamic_QA"] = """

You are an AI assistant that answers questions based on both temporal event sequences and relevant text chunks.

# TASK
Answer the question using both the provided events and text chunks. The events show temporal relationships, while the text chunks provide additional context and details.

# TEMPORAL RELEVANCE GUIDELINE
**Prioritize information from time periods that align with the question's temporal scope.** If a question asks about a specific time period, focus on events and information that occurred within or near that timeframe. Consider whether nearby temporal information provides relevant context or background that could help answer the question.

# ANALYSIS APPROACH
1. **Query Semantics Understanding**: Analyze the semantic intent of the question:
   - **Existential queries**: Questions about what exists/happens at a specific time
   - **Continuity queries**: Questions about ongoing states, processes, or relationships
   - **Boundary queries**: Questions about beginnings, endings, or transitions
   - **Aggregate queries**: Questions requiring synthesis across multiple time points

2. **Temporal Scope Identification**: Identify the exact time constraints in the question:
   - **Subject Consistency Check**: Verify that the event's subject matches the question's focus
   - Extract specific dates, time periods, or temporal references
   - Mark any temporal qualifiers (e.g., "during", "between", "before", "after")

3. **Evidence Time Filtering**: Before using any event or chunk as evidence:
   - **Assess temporal relevance**: Evaluate whether each piece of evidence falls within or near the question's time scope
   - **Consider contextual value**: Include information from nearby time periods if it provides essential background
   - **Prioritize temporal proximity**: Give preference to evidence closer to the target time period

4. **Event Analysis**: Review the temporal sequence of events:
   - **Temporal Ordering**: Analyze when events happen and their chronological relationships
   - **State Persistence and Change**: 
     - **Persistent states**: Properties or conditions that continue until explicitly changed
     - **Instantaneous events**: Actions that happen at specific moments
     - **Processes**: Ongoing activities with duration
     - **Transitions**: Changes from one state to another
   - **Office/Role Incumbency Reasoning**: For questions about who held a position at a given time:
     - Treat holding an office as a persistent state that starts at appointment and lasts until explicit end
     - Choose the most recent appointment before or at the query time that has no earlier termination
     - If no explicit end event is found before the query time, assume the office holder remains in position

5. **Entity-Event Relationships**: Analyze connections between entities and events:
   - **Agent relationships**: Who performs or causes an action
   - **Patient relationships**: Who/what is affected by an action
   - **Locative relationships**: Where something happens
   - **Attributive relationships**: Properties or characteristics at specific times

6. **Chunk Analysis**: Extract relevant information from the text chunks that supplements the events

7. **Cross-Reference**: Connect information between events and chunks to provide a comprehensive answer

# RESPONSE GUIDELINES
- **TEMPORAL RELEVANCE ASSESSMENT**: Assess whether evidence timestamps are relevant to the question's time scope
- Reference specific events (e.g., "Event #3") or chunks when supporting your answer
- If information is available in both events and chunks, prioritize the most specific and temporally relevant details
- Apply temporal logic: if an event occurred at time T1 and no change is mentioned by T2, assume continuity
- **If limited information exists for the queried time period**: 
  - State the temporal limitations and explain how nearby temporal information helps provide context
  - Make reasonable inferences based on:
    1. **Career Continuity**: If a person's career shows clear progression in a field/organization, infer likely continuity
    2. **Role Persistence**: If someone holds a position before and after the queried time, infer likely continued role
    3. **Institutional Affiliation**: If consistently associated with an institution, infer likely continued affiliation
  - Label inferences clearly and explain the reasoning chain
- Start with the direct answer followed by justification citing key events
- If the answer is based on inference, state: "Based on inference from surrounding evidence: [answer]" and explain the reasoning

# QUESTION
{question}

# EVENTS
{events_data}

# TEXT CHUNKS
{chunks_data}

# ANSWER
"""

PROMPTS["dynamic_QA_wo_timeline"] = """You are an AI assistant that answers questions based on relevant text chunks.

# TASK
Answer the question using the provided text chunks to provide additional context and details.

# CRITICAL TEMPORAL CONSTRAINT RULE
**NEVER use information from time periods that do not match the question's temporal scope.** If a question asks about a specific time period (e.g., "between 2000-2001"), only use information that occurred within or are explicitly valid for that exact time range. Using information from different time periods (e.g., 2009-2012 data for a 2000-2001 question) is a critical error.

# ANALYSIS APPROACH
1. **Query Semantics Understanding**: Analyze the semantic intent of the question:
   - **Existential queries**: Questions about what exists/happens at a specific time
   - **Causal queries**: Questions about causes, effects, or consequences over time
   - **Comparative queries**: Questions comparing states across different time periods
   - **Continuity queries**: Questions about ongoing states, processes, or relationships
   - **Boundary queries**: Questions about beginnings, endings, or transitions
   - **Aggregate queries**: Questions requiring synthesis across multiple time points

2. **Temporal Scope Identification**: FIRST identify the exact time constraints in the question:
   - Extract specific dates, time periods, or temporal references
   - Determine the precise temporal boundaries for valid evidence
   - Mark any temporal qualifiers (e.g., "during", "between", "before", "after")

3. **Evidence Time Filtering**: Before using any chunk as evidence:
   - **Verify temporal relevance**: Check that each piece of evidence falls within the question's time scope
   - **Reject out-of-scope evidence**: Discard any information from time periods outside the question's temporal constraints
   - **Flag temporal mismatches**: Identify when available evidence doesn't match the queried time period

4. **Chunk Analysis**: Extract relevant information from the text chunks

5. **Temporal Reasoning Strategies**: Apply sophisticated temporal logic:
   - **Forward chaining**: From past information, infer current states
   - **Backward chaining**: From current states, infer necessary past information
   - **Interval reasoning**: For queries about time periods, consider all relevant information within and bounding the interval
   - **Default persistence**: If X was true at time T1 and no change is mentioned by T2, assume X remains true at T2
   - **Temporal granularity matching**: Align the precision of your answer with the query's temporal specificity

# RESPONSE GUIDELINES
- **MANDATORY TEMPORAL VERIFICATION**: Before citing any evidence, verify its timestamps match the question's time scope
- Reference specific chunks when supporting your answer, but ONLY if they are from the correct time period
- Prioritize the most specific and relevant details from the correct time period
- Distinguish between definite facts and probable inferences
- **If no information exists for the queried time period**: Explicitly state "No information is available for the specified time period" rather than using irrelevant temporal data
- When uncertain, clearly state the limitations of the available information

# QUESTION
{question}

# TEXT CHUNKS
{chunks_data}

# ANSWER
"""

PROMPTS["extract_2_step_events"]="""
You are tasked with extracting events from the given content.

Here's the outline of what you need to do:

Information-value Filter
- For each candidate sentence, compute an information score (0-4):
  1. **Specific actor**: +1 point if contains a named entity or definite noun phrase uniquely identifying who acted
  2. **Action/Change**: +1 point if describes an action, role/office held, or continued state with temporal boundaries
  3. **Result/magnitude**: +1 point if includes quantitative detail, result, or consequence
  4. **Temporal anchoring**: +1 point if time can be resolved to at least month precision

- Keep the sentence if:
  - Score ≥ 1, OR
  - Matches pattern: <Actor> was/served as <Role> at/for <Org> from <Start> to/until <End>, OR
  - Contains explicit temporal markers with identifiable subject and basic action

Events Extraction Rules
1. Determine time type:
   - time_point: specific point in time (set start_time, leave end_time empty)
   - time_interval: start and end time (set both start_time and end_time)

2. For each event:
   - event_id: "E" + number (e.g., E1, E2)
   - sentence: Original sentence with complete temporal information and explicit subject (no pronouns)
   - context: Optional ≤80 tokens supplementary information from same text chunk
   - start_time: Start time in format (YYYY,MM,DD) or (YYYY,,) or (YYYY,MM,) 
   - end_time: End time (empty for time_point events)
   - time_static: "True" if no temporal message, else "False"

Temporal Handling
- Single time points: Set both start_time and end_time to cover that period
- Directional expressions:
  - "after 2020" → start_time: "(2020,,)", end_time: null
  - "before 2020" → start_time: null, end_time: "(2020,,)"
  - "since 2020" → start_time: "(2020,,)", end_time: null
- Relative times: Convert to absolute dates based on context
- Time ranges: "1990s" → start: "(1990,01,01)", end: "(1999,12,31)"
- Multiple time points in one sentence: Split into multiple events
- DO NOT merge events on different days

Critical Requirements
- Every sentence MUST have clear, explicit subject (NO pronouns)
- Use complete names/titles (not abbreviations or nicknames)
- Scan entire document to resolve pronoun references
- If unsure or cannot resolve pronouns, output {"events": []}
- Time expressions must appear unaltered in sentence or context

Notes:
- If Action / Change is present but Temporal anchoring is not, you must attempt read the context to find the time; if still unresolved → discard.
- Permanent attributes (citizenship, birthplace, chemical formula…) may remain with time_static: true only when they lack action verbs.
- For single time points in sentence (like "in 2010" or "May 2023"):
  * If the context implies events AT or DURING this time, set both start_time and end_time to cover that specific period
  * If the context implies events AFTER this time, set start_time to this time and end_time to null
  * If the context implies events BEFORE this time, set start_time to null and end_time to this time
- For directional time expressions:
  * "after 2020" → set start_time: "(2020,,)", end_time: null
  * "before 2020" → set start_time: null, end_time: "(2020,,)"  
  * "since 2020" → set start_time: "(2020,,)", end_time: null
- For relative times (like "last week"), convert to absolute dates based on the temporl messages in context
- For time expressions like "in the 1990s", use the appropriate start and end dates ((1990,01,01) and (1999,12,31))
- Modify temporal messages to the corresponding standardized formatafter above steps.
- The time_interval for a relationship must accurately represent the primary temporal scope described by the relationship's core meaning.
- When describing a state, role, or performance (e.g., "was a regular", "served as", "worked during"), the time_interval should cover the entire duration of that state or role.
- Other temporal details mentioned in the sentence (e.g., periods introduced by "except for", "other than", "from...to...") are used to qualify or contrast the main state, but do not redefine the primary time_interval. Mention these qualifying periods in the description field if necessary.
- If the primary temporal scope is a named, well-defined period (e.g., a tenure, a sports season, a fiscal year), use the standardized start and end dates for that period.
- When a sentence contains both a precise point event (e.g., signing date) and a period range (e.g., a season), determine which one to use as the time_interval based on the semantic meaning of the relationship type.
- If there are a few time points in one sentence, split them into multiple events.
- Do NOT merge events that occur on different days.
- The explicit or interval expression that anchors the time must appear unaltered in either sentence or context so that downstream models can recover the original phrasing.
- Every extracted sentence MUST have a clear, explicit subject - no subject-less sentences are allowed.
- Use the most complete form available - prefer full names over nicknames, official titles over informal references, complete organization names over abbreviations.
- ABSOLUTELY NO pronouns like "he", "she", "they", "it", "this", "that" are permitted in the extracted sentences.
- Scan the entire document (include title,not just the immediate context) to identify the full name or complete designation of any person, organization, or entity.
- The explicit or interval expression that anchors the time must appear unaltered in either sentence or context so that downstream models can recover the original phrasing.
- If the chunk contains no events or cannot resolve all pronouns to specific names, output { "events": [] }.


Example:
Input:
Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019. During that four-year period, her team published over 20 papers and secured two major grants (2017, 2018). She continued her advisory role for the project until its official conclusion in 2022.

Output:
{
  "events": [
    {
      "event_id": "E1",
      "sentence": "Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019.",
      "context": "This period lasted four years.",
      "start_time": "(2015,,)",
      "end_time": "(2019,,)",
      "time_static": false
    },
    {
      "event_id": "E2",
      "sentence": "Dr. Elena Martinez's team secured a major grant in 2017.",
      "context": "This was one of two major grants secured during the Pacific Currents Project; the team secured over 20 papers and two major grants from 2015 to 2019.",
      "start_time": "(2017,,)",
      "end_time": "",
      "time_static": false
    },
    {
      "event_id": "E3",
      "sentence": "Dr. Elena Martinez's team secured a major grant in 2018.",
      "context": "This was one of two major grants secured during the Pacific Currents Project; the team secured over 20 papers and two major grants from 2015 to 2019.",
      "start_time": "(2018,,)",
      "end_time": "",
      "time_static": false
    },
    {
      "event_id": "E4",
      "sentence": "Dr. Elena Martinez continued her advisory role for the Oceanography Institute's Pacific Currents Project until its official conclusion in 2022.",
      "context": "Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019.",
      "start_time": "(2019,,)",
      "end_time": "(2022,,)",
      "time_static": false
    }
  ]
}
Input:
{input_text}


"""

PROMPTS["extract_2_step_entities"]="""
You are tasked with extracting entities which can be retrived directly from given event list.

Here’s the outline of what you need to do:

Entity extractions

Each identified entity should have a unique identifier (id),a type (type),a event_id that marked the entity is extract from which event,and a description：
- Entity types must be specific and precise.
- If the entity can be extracted from more than one events,link all the ids of the events by <SEP>.(e.g.,E1<SEP>E2)
- Descriptions must have strong specificity and must be structured as a single, coherent sentence that naturally incorporates:Key distinguishing features (unique attributes that differentiate it from similar entities).Specific role or function within the given context.Relevant temporal, spatial, or domain-limiting information.Clear relationships to other mentioned entities when applicable

Notes:
- Use the most complete form available - prefer full names over nicknames, official titles over informal references, complete organization names over abbreviations.
- ABSOLUTELY NO pronouns like "he", "she", "they", "it", "this", "that" are permitted in the extracted sentences.

Example 

Input:

{
  "events": [
    {
      "event_id": "E1",
      "sentence": "Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019.",
      "context": "This period lasted four years.",
      "start_time": "(2015,,)",
      "end_time": "(2019,,)",
      "time_static": false
    },
    {
      "event_id": "E2",
      "sentence": "Dr. Elena Martinez's team secured a major grant in 2017.",
      "context": "This was one of two major grants secured during the Pacific Currents Project; the team secured over 20 papers and two major grants from 2015 to 2019.",
      "start_time": "(2017,,)",
      "end_time": "",
      "time_static": false
    },
    {
      "event_id": "E3",
      "sentence": "Dr. Elena Martinez's team secured a major grant in 2018.",
      "context": "This was one of two major grants secured during the Pacific Currents Project; the team secured over 20 papers and two major grants from 2015 to 2019.",
      "start_time": "(2018,,)",
      "end_time": "",
      "time_static": false
    },
    {
      "event_id": "E4",
      "sentence": "Dr. Elena Martinez continued her advisory role for the Oceanography Institute's Pacific Currents Project until its official conclusion in 2022.",
      "context": "Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019.",
      "start_time": "(2019,,)",
      "end_time": "(2022,,)",
      "time_static": false
    }
  ]
}
    
Output:

{
  "entities": [
    {
      "id": "Dr. Elena Martinez",
      "type": "person",
      "event_id": "E1<SEP>E2<SEP>E3<SEP>E4",
      "description": "Dr. Elena Martinez was the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 to 2019, during which her team published over 20 papers and secured two major grants; she later served in an advisory role for the project until its conclusion in 2022."
    },
    {
      "id": "Oceanography Institute's Pacific Currents Project",
      "type": "research_project",
      "event_id": "E1<SEP>E4",
      "description": "The Oceanography Institute's Pacific Currents Project was a research initiative led by Dr. Elena Martinez from 2015 to 2019, which produced over 20 papers and secured two major grants, and concluded officially in 2022."
    }
  ]
}

Notice

If you are unsure about extracting any entity or cannot resolve all pronouns to specific names, output { "entities": [] }.
Do not reveal your internal scoring or reasoning—only return the final JSON.

Real-Data
Input: {input_text}
Output:
"""

PROMPTS["dynamic_event_units"] = """You are tasked with extracting events and nodes which can be retrived directly from given content.

Here’s the outline of what you need to do:

Information-value filter

-For each candidate sentence, compute an information score (0–4):
* Criterion: Specific actor
* +1 point if: Contains a named entity or definite noun phrase uniquely identifying who acted.
* Rationale: Avoids generic "someone"/"people".

* Criterion: Action / Change
* +1 point if: Describes an action or a clearly stated role/office held (served as, was appointed) or a continued state/membership with temporal boundaries (remained as, continued until).
* Rationale: Captures dynamic facts, milestones, and significant durations.

* Criterion: Result / magnitude
* +1 point if: Includes a quantitative detail, result, or consequence ($5 M, two satellites).
* Rationale: Adds substantive content.

* Criterion: Temporal anchoring
* +1 point if: Time can be resolved to at least month precision, or has a clear temporal boundary (start point, end point, or bounded interval).
* Rationale: Ensures usefulness for temporal queries.

* Keep the sentence if score ≥ 1 OR it matches the pattern <Actor> was/served as <Role> at/for <Org> from <Start> to / until <End> OR it contains explicit temporal markers (bracketed years, standalone sentence-ending years, or clear date intervals) with identifiable subject and basic action OR it describes a continued state/membership with clear temporal boundaries (remained until, continued as, stayed as).

Events extraction

Only extract events from the sentence that are kept after Information-value filter.
First, determine whether the time information contained in the event is a time_point or a time_interval, as defined below:
* time_point: The specific point in time when the event occurred.If the temporal_message is a time_point,set the start_time as the time_point,and set leave the end_time empty.
* time_interval: The start and end time of the event. If the temporal_message is a time_point,set the start_time as the start of the time_interval,and set the end_time as the end of the time_interval.
and 
Each relationship must have:
- event_id: linked by 'E' and the number of the event in the event list(e.g.,E1,E2...)
- sentence: The original sentence from the input content from which this event and the relational information was extracted.If the event is extracted from two or more sentences, merge them into a single.The sentence must contains all the temporal information and self-contained(or semicolon-joined compound) ,including the key fact, retains the original time expression, and has a complete, explicit subject with no pronouns.
- context: optionally supply ≤ 80 tokens of supplementary information (background, consequence, aliases, quantitative details) drawn only from the same text chunk.
- start_time: The start time of the event,if the event has no temporal message,leave this field empty.
- end_time: The end time of the event,if the event has no temporal message or the temporal message of the event is a time_point,leave this field empty.
- time_static: If the event has no temporal message,set this field as 'True',else,set this field as 'Fales'.

Entity extractions

Each identified entity should have a unique identifier (id),a type (type),a event_id that marked the entity is extract from which event,and a description：
- Entity types must be specific and precise.
- If the entity can be extracted from more than one events,link all the ids of the events by <SEP>.(e.g.,E1<SEP>E2)
- Descriptions must have strong specificity and must be structured as a single, coherent sentence that naturally incorporates:Key distinguishing features (unique attributes that differentiate it from similar entities).Specific role or function within the given context.Relevant temporal, spatial, or domain-limiting information.Clear relationships to other mentioned entities when applicable

Notes:
- If Action / Change is present but Temporal anchoring is not, you must attempt read the context to find the time; if still unresolved → discard.
- Permanent attributes (citizenship, birthplace, chemical formula…) may remain with time_static: true only when they lack action verbs.
- For single time points in sentence (like "in 2010" or "May 2023"):
  * If the context implies events AT or DURING this time, set both start_time and end_time to cover that specific period
  * If the context implies events AFTER this time, set start_time to this time and end_time to null
  * If the context implies events BEFORE this time, set start_time to null and end_time to this time
- For directional time expressions:
  * "after 2020" → set start_time: "(2020,,)", end_time: null
  * "before 2020" → set start_time: null, end_time: "(2020,,)"  
  * "since 2020" → set start_time: "(2020,,)", end_time: null
- For relative times (like "last week"), convert to absolute dates based on the temporl messages in context
- For time expressions like "in the 1990s", use the appropriate start and end dates ((1990,01,01) and (1999,12,31))
- Modify temporal messages to the corresponding standardized formatafter above steps.
- The time_interval for a relationship must accurately represent the primary temporal scope described by the relationship's core meaning.
- When describing a state, role, or performance (e.g., "was a regular", "served as", "worked during"), the time_interval should cover the entire duration of that state or role.
- Other temporal details mentioned in the sentence (e.g., periods introduced by "except for", "other than", "from...to...") are used to qualify or contrast the main state, but do not redefine the primary time_interval. Mention these qualifying periods in the description field if necessary.
- If the primary temporal scope is a named, well-defined period (e.g., a tenure, a sports season, a fiscal year), use the standardized start and end dates for that period.
- When a sentence contains both a precise point event (e.g., signing date) and a period range (e.g., a season), determine which one to use as the time_interval based on the semantic meaning of the relationship type.
- If there are a few time points in one sentence, split them into multiple events.
- Do NOT merge events that occur on different days.
- The explicit or interval expression that anchors the time must appear unaltered in either sentence or context so that downstream models can recover the original phrasing.
- Every extracted sentence MUST have a clear, explicit subject - no subject-less sentences are allowed.
- Use the most complete form available - prefer full names over nicknames, official titles over informal references, complete organization names over abbreviations.
- ABSOLUTELY NO pronouns like "he", "she", "they", "it", "this", "that" are permitted in the extracted sentences.
- Scan the entire document (include title,not just the immediate context) to identify the full name or complete designation of any person, organization, or entity.
- The explicit or interval expression that anchors the time must appear unaltered in either sentence or context so that downstream models can recover the original phrasing.
- If the chunk contains no events or cannot resolve all pronouns to specific names, output { "events": [] }.

Example 1

Input:
Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019. During that four-year period, her team published over 20 papers and secured two major grants (2017, 2018). She continued her advisory role for the project until its official conclusion in 2022.

Output:

{
  "events": [
    {
      "event_id": "E1",
      "sentence": "Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019.",
      "context": "This period lasted four years.",
      "start_time": "(2015,,)",
      "end_time": "(2019,,)",
      "time_static": false
    },
    {
      "event_id": "E2",
      "sentence": "Dr. Elena Martinez's team secured a major grant in 2017.",
      "context": "This was one of two major grants secured during the Pacific Currents Project; the team secured over 20 papers and two major grants from 2015 to 2019.",
      "start_time": "(2017,,)",
      "end_time": "",
      "time_static": false
    },
    {
      "event_id": "E3",
      "sentence": "Dr. Elena Martinez's team secured a major grant in 2018.",
      "context": "This was one of two major grants secured during the Pacific Currents Project; the team secured over 20 papers and two major grants from 2015 to 2019.",
      "start_time": "(2018,,)",
      "end_time": "",
      "time_static": false
    },
    {
      "event_id": "E4",
      "sentence": "Dr. Elena Martinez continued her advisory role for the Oceanography Institute's Pacific Currents Project until its official conclusion in 2022.",
      "context": "Dr. Elena Martinez served as the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 until she took a sabbatical in 2019.",
      "start_time": "(2019,,)",
      "end_time": "(2022,,)",
      "time_static": false
    }
  ],
  "entities": [
    {
      "id": "Dr. Elena Martinez",
      "type": "person",
      "event_id": "E1<SEP>E2<SEP>E3<SEP>E4",
      "description": "Dr. Elena Martinez was the lead researcher for the Oceanography Institute's Pacific Currents Project from 2015 to 2019, during which her team published over 20 papers and secured two major grants; she later served in an advisory role for the project until its conclusion in 2022."
    },
    {
      "id": "Oceanography Institute's Pacific Currents Project",
      "type": "research_project",
      "event_id": "E1<SEP>E4",
      "description": "The Oceanography Institute's Pacific Currents Project was a research initiative led by Dr. Elena Martinez from 2015 to 2019, which produced over 20 papers and secured two major grants, and concluded officially in 2022."
    }
  ]
}

Notice

If you are unsure about extracting any event or cannot resolve all pronouns to specific names, output { "events": [] }.
Do not reveal your internal scoring or reasoning—only return the final JSON.
REMEMBER: Keep the sentence information as complete and as accurate as possible to be retrieved directly for event envidence retrieval.

Real-Data
Input: 
{input_text}
Output:
"""


PROMPTS["time_entity_extraction"] = """You are an expert at analyzing queries to extract temporal constraints and named entities.

Given the following query: "{query}"

Extract:
1. Time constraints: Any temporal constraints mentioned in the query, including explicit dates, relative time references, or time periods.
2. Entities: Any named entities mentioned in the query that the user might be interested in.

Please return your analysis in JSON format with the following structure:
{
  "time_constraints": {
    "start_time": "YYYY-MM-DD or null if not mentioned", 
    "end_time": "YYYY-MM-DD or null if not mentioned"
  },
  "entities": ["entity1", "entity2", ...]
}
```

Notes:
- For time constraints, normalize to ISO date formats (YYYY-MM-DD) when possible
- For single time points (like "in 2010" or "May 2023"):
  * If the context implies events AT or DURING this time, set both start_time and end_time to cover that specific period
  * If the context implies events AFTER this time, set start_time to this time and end_time to null
  * If the context implies events BEFORE this time, set start_time to null and end_time to this time
- For directional time expressions:
  * "after 2020" → set start_time: "2020-01-01", end_time: null
  * "before 2020" → set start_time: null, end_time: "2020-01-01"  
  * "since 2020" → set start_time: "2020-01-01", end_time: null
- For relative times (like "last week"), convert to absolute dates based on the current date
- For time expressions like "in the 1990s", use the appropriate start and end dates (1990-01-01 and 1999-12-31)
- For entities, extract proper nouns, organizations, people, and key concepts
- Use null for time values that aren't specified

Current date for reference: {current_date}

{examples}

JSON Response:
"""

PROMPTS["time_entity_extraction_examples"] = [
    """Example 1:
Query: "What happened to Apple stock between January 2020 and March 2021?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2020-01-01",
    "end_time": "2021-03-31"
  },
  "entities": ["Apple", "stock"]
}
```""",

    """Example 2:
Query: "Tell me about Microsoft's acquisitions in the last decade"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2013-01-01",
    "end_time": "2023-12-31"
  },
  "entities": ["Microsoft", "acquisitions"]
}
```""",

    """Example 3:
Query: "What were Tesla's major challenges in 2022?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2022-01-01",
    "end_time": "2022-12-31"
  },
  "entities": ["Tesla", "challenges"]
}
```""",

    """Example 4:
Query: "How did COVID-19 affect global economies?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2019-12-01",
    "end_time": null
  },
  "entities": ["COVID-19", "global economies"]
}
```""",

    """Example 5:
Query: "What happened after Google's IPO in August 2004?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2004-08-01", 
    "end_time": null
  },
  "entities": ["Google", "IPO"]
}
```""",

    """Example 6:
Query: "Tell me about events before the 2008 financial crisis."
JSON Response:
```json
{
  "time_constraints": {
    "start_time": null, 
    "end_time": "2008-09-15"
  },
  "entities": ["financial crisis"]
}
```""",

    """Example 7:
Query: "What innovations were made in May 2019?"
JSON Response:
```json
{
  "time_constraints": {
    "start_time": "2019-05-01", 
    "end_time": "2019-05-31"
  },
  "entities": ["innovations"]
}
```"""
]

PROMPTS[
    "event_continue_extraction"
] = """MANY events and entities were missed in the last extraction. Add them below using the same format with an "events" list and an "entities" list,only out put the missing events and entities.
"""

PROMPTS[
    "eventonly_continue_extraction"
] = """MANY events  were missed in the last extraction. Add them below using the same format with an "events" list ,only out put the missing events .
"""


PROMPTS[
    "entityonly_continue_extraction"
] = """MANY entities were missed in the last extraction. Add them below using the same format with an "entities" list,only out put the missing entities.
"""


PROMPTS[
    "event_if_loop_extraction"
] = """It appears some events and entities may have still been missed. Answer YES | NO if there are still events or entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]



PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
