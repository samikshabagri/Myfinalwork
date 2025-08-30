# Cover &#x26; How to Use

# Maritime Virtual Assistant — Test Pack (Synthetic Data)

What this is:

A self-contained, fictional set of voyage docs to test an AI agent that can:

- Answer laytime/weather/distance/CP clause queries
- Retrieve &#x26; summarize uploaded docs
- Suggest actions &#x26; documents per voyage stage
- Support voice/chat interaction

How to test:

1. Upload this PDF to your agent.
2. Ask questions such as:
3. - "Summarize the Charter Party laytime terms and demurrage."
- "Compute laytime used from the SOF and say if demurrage/despatch applies."
- "What documents are required before berthing?"
- "What’s the distance and ETA Singapore → Fujairah at 13 knots?"
- "Was any time excepted due to weather or holidays?"

Ask the agent to produce a checklist for each voyage stage.
4. Ask it to extract key data points (vessel, cargo, laycan, port names, quantities).

Note: All figures and clauses here are synthetic and not real-world advice.


---


# Fixture Recap &#x26; CP Clauses (Excerpt)

Vessel: M/V OCEAN SWIFT (Handysize, 32,500 DWT)

Cargo: 30,000 MT Wheat in bulk, 10% more/less at Charterers' option

Load Port: Singapore, PSA (1 SB)

Discharge Port: Fujairah, Oil Terminal (1 SB)

Laycan: 20–22 July 2025

Speed/Consumption (WOG): 13.0 kn on 22.5 mt IFO + 1.5 mt MGO/day in good weather

Freight: USD 29.50/MT FIOST

Demurrage: USD 15,000/day, pro rata

Despatch: USD 7,500/day, pro rata (half demurrage)

# Laytime Terms:

- Total Laytime Allowed: 72 hours SHEX EIU (Sundays &#x26; Holidays Excepted, Even If Used)
- Notice of Readiness (NOR): To be tendered upon arrival at pilot station, WIPON/WIBON/WCCON
- Laytime Commencement: 6 hours after valid NOR or upon berthing &#x26; readiness, whichever earlier
- Time Counting: SHEX EIU; rain stoppages and shifting time between berths not to count
- Weather Delays: Time lost due to weather at roadstead/berth to be excepted
- Shifting: Time used for shifting between berths not to count
- Draft/Docs: Master to present original Bills of Lading and cargo documents as required
- Canal Transit: If any canals transited, time counts as used for the voyage (not laytime)

# Other CP Clauses:

- Cargo Gear: Vessel gear available, 4x30T cranes, grabs by shore
- CP Law &#x26; Jurisdiction: English law, London arbitration
- Deviation: Reasonable deviation permitted for safety and bunkering


---



Statement of Facts (SOF)

# Port: Singapore (Load) — Terminal: PSA

# Vessel: M/V OCEAN SWIFT

# Agent: Fairwinds Shipping

# Dates: 21–24 July 2025 (Local Time)

# 21 Jul 2025

05:40 Arrived pilot station, NOR tendered WIPON/WIBON/WCCON

06:00 Pilot on board

06:35 All fast, alongside Berth L-3

07:10 Commenced pre-cargo checks

08:00 Commenced loading

12:45 Rain started — loading suspended

14:10 Rain ceased — loading resumed

# 22 Jul 2025

00:00 Loading continues

02:30 Shore breakdown — loading stopped

03:15 Shore fixed — loading resumed

13:20 Shifting ordered to Berth L-5

14:00 Let go — shifting commences

14:45 All fast at L-5 — hose/gear connections

15:10 Loading resumed

# 23 Jul 2025

09:30 Loading completed, tally confirmation

10:00 Draft survey commences

12:15 Draft survey completed

13:00 Documents prepared/signing

15:40 All docs on board, laytime to cease per valid completion

# Cargo Quantity

29,700 MT loaded

# Weather Events

Rain (21 Jul 12:45–14:10)

# Shifting Time

22 Jul 14:00–14:45

# Excepted Time

Rain stoppage, shifting time (per CP)




---



# Weather Extract (Synthetic)

Area: Singapore — July 2025

General: SW monsoon pattern, scattered showers, isolated thunderstorms late morning

# METAR (Illustrative):

WSSS 211200Z 23008KT 9999 -RA FEW018 SCT025 BKN080 29/25 Q1006

# Forecast Window:

Intermittent showers 11:30–14:30 LT on 21 Jul, clearing thereafter

# Sea State:

Slight to moderate, short-period swell

# Operational Guidance:

- Expect brief rain delays around midday
- Lightning risk advisory may suspend crane ops for 20–40 mins




---


# Distances &#x26; ETA (Synthetic)

# Approximate Great-Circle Distances (NM):

- Singapore → Fujairah: 3,300 NM
- Singapore → Colombo: 1,550 NM
- Colombo → Fujairah: 1,780 NM

# ETA Calculation Template:

Given speed V (kn) and distance D (NM):

Transit Time (days) = D / (24 * V)

# Example (Singapore → Fujairah @ 13 kn):

D = 3,300 NM

Time = 3,300 / (24 * 13) ≈ 10.58 days

If departing 24 Jul 2025 18:00 LT → ETA ≈ 04 Aug 2025 07:55 LT

(Note: Synthetic values; adjust for weather/traffic/canals as needed)


---


# Voyage Stage Checklist

# Pre-Fixture:

- Validate laycan feasibility vs distance/speed/port congestion
- Review CP terms: laytime, NOR, SHEX/SHINC, demurrage/despatch
- Confirm cargo readiness &#x26; docs list

# Pre-Arrival:

- 96/72/48/24-hr notices to agent/terminal
- Crew brief on NOR requirements and readiness
- Prepare draft surveyors and tally arrangements

# In-Port:

- Tender NOR per CP (WIPON/WIBON/WCCON as applicable)
- Log all stoppages with reasons (rain, shore breakdown, shifting)
- Keep SOF updated and signed by parties

# Post-Departure / Post-Operation:

- Finalize SOF, draft surveys, B/Ls, Mate’s Receipt
- Laytime statement preparation &#x26; settlement
- File weather logs, ECDIS tracks (if needed)


---


# Laytime Worksheet (To Be Computed by Agent)

Laytime Allowed: 72 hours SHEX EIU

# Key Events (per SOF, local time):

- NOR tendered: 21 Jul 05:40
- All fast: 21 Jul 06:35
- Loading start: 21 Jul 08:00
- Rain stop: 21 Jul 12:45 — resume 14:10 (EXCEPTED)
- Shore breakdown: 22 Jul 02:30 — resume 03:15 (COUNTS per CP unless otherwise stated)
- Shifting: 22 Jul 14:00–14:45 (EXCEPTED)
- Loading complete: 23 Jul 09:30
- Laytime ends: 23 Jul 15:40 (Docs finalized per CP completion)

Assume: Sundays &#x26; Holidays Excepted Even If Used (SHEX EIU), no official holidays during window.

# Task:

1. Determine laytime commencement (NOR vs berthing rule: 6 hours after valid NOR or upon berthing &#x26; readiness, whichever earlier).
2. Subtract excepted periods (rain + shifting).
3. Count any shore breakdown per CP terms (this CP counts it).
4. Compute total laytime used vs 72 hours.
5. Decide demurrage or despatch and amount at USD 15,000/day or USD 7,500/day (pro rata).


---


# Sample Clauses &#x26; Glossary

# Abbreviations:

- WIPON: Whether In Port or Not
- WIBON: Whether In Berth or Not
- WCCON: Whether Customs Cleared or Not
- SHEX EIU: Sundays &#x26; Holidays Excepted, Even If Used
- SHINC: Sundays &#x26; Holidays Included

# Sample CP Snippets (illustrative):

- NOR shall be tendered at pilot station and shall be valid when the Vessel is in all respects ready to load/discharge.
- Laytime shall commence 6 hours after valid NOR or upon berthing and readiness, whichever earlier.
- Time lost due to shifting between berths shall not count as laytime.
- Time lost due to weather preventing operations shall not count.
- Shore equipment breakdown shall count as laytime.


---


# Prompt Ideas

Try these prompts with your assistant:

- "Summarize all laytime-related clauses from this document."
- "Create a structured JSON of key voyage data (vessel, ports, laycan, cargo, rates)."
- "Compute laytime used and state demurrage/despatch amount."
- "List the excepted periods and justify using CP language."
- "Estimate ETA from Singapore to Fujairah at 13 kn departing 24 Jul 2025 18:00."
- "Generate a pre-arrival checklist based on the CP and SOF."
- "Identify missing documents needed before sailing."
