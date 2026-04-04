# AI-Based Driver Drowsiness Detection

## Complete Project Guide (For Viva + Report)

## 1. Project Summary
This project started as a Computer Vision system that only detected drowsiness using eye closure from OpenCV/MediaPipe.

Now it has been upgraded into an AI Lab style project by adding:
- A rule-based expert system (knowledge base)
- An inference engine (forward chaining)
- State space search over fatigue states
- A heuristic function for fast decision making
- A decision output layer with different actions per state

So the system is no longer only detection-based. It now performs AI reasoning before deciding the final driver condition.

## 2. Problem Statement
Simple detection systems usually use one condition:
- "If eyes are closed for long enough, then drowsy"

This is limited and not explainable enough for AI Lab requirements.

Our upgraded system solves this by:
- Using multiple rules
- Maintaining state transitions
- Explaining the matched rules and final decision
- Providing measurable heuristic score

## 3. Objective
To convert a pure vision project into an AI-driven decision-making system that can:
- Infer driver state using rules
- Search among possible states
- Use heuristic thresholds for practical reasoning
- Trigger actions based on inferred state severity

## 4. System Architecture
The processing pipeline is:

1. Camera frame capture
2. Face + eye landmark extraction (MediaPipe)
3. EAR (Eye Aspect Ratio) calculation
4. Feature extraction:
- eye closure duration (seconds)
- blink rate (blinks per minute)
5. AI inference layer:
- Knowledge base rule matching
- Forward chaining inference
- Heuristic-based state search
6. Final decision state output
7. Action layer (warning/alarm)

## 5. Knowledge Base (Rules Used)
The rule base implemented is:

- R1: IF eyes_closed_time < 3 sec, THEN state = ALERT
- R2: IF eyes_closed_time between 3 and 5 sec, THEN state = DROWSY
- R3: IF eyes_closed_time > 5 sec, THEN state = CRITICAL
- R4: IF blink_rate < tired_threshold, THEN state = TIRED

Default threshold values:
- alert_max_closed_seconds = 3.0
- drowsy_min_closed_seconds = 3.0
- critical_min_closed_seconds = 5.0
- tired_blink_rate_threshold = 12.0 blinks/min
- blink_window_seconds = 60.0

These thresholds are configurable from the frontend and API.

## 6. Inference Engine (Forward Chaining)
Forward chaining means we start with known facts (input measurements), then apply rules to infer output.

Facts collected per frame:
- closed_seconds
- blink_rate_per_min

Inference steps:
1. Read current facts.
2. Check each rule condition in sequence.
3. Add matching rules to matched_rules list.
4. Generate candidate states from matched conditions.
5. Select best final state using heuristic state search.

This gives explainable AI behavior because we can show exactly which rules fired.

## 7. State Space Representation
State space used:
- ALERT
- TIRED
- DROWSY
- CRITICAL

The engine searches this state space and picks the best state based on:
- Rule constraints (allowed states)
- Heuristic closeness (fatigue score vs state target)

State anchors used for search:
- ALERT -> 0.0
- TIRED -> 0.35
- DROWSY -> 0.7
- CRITICAL -> 1.0

## 8. Heuristic Function
A fatigue heuristic score in range [0,1] is computed:

- closure_score = closed_seconds / critical_min_closed_seconds (clipped to [0,1])
- blink_score = (tired_threshold - blink_rate) / tired_threshold (clipped to [0,1])
- heuristic_score = 0.7 * closure_score + 0.3 * blink_score

Why this heuristic?
- Eye closure duration is stronger indicator, so it has higher weight (0.7).
- Low blink rate is additional support, so lower weight (0.3).

This makes decision practical and fast for real-time use.

## 9. Decision Output Layer
Final action mapping:

- ALERT -> NO_ACTION
- TIRED -> DISPLAY_WARNING
- DROWSY -> SOUND_ALARM
- CRITICAL -> CONTINUOUS_ALERT

Behavior in project:
- TIRED: visual warning state only
- DROWSY: alarm starts
- CRITICAL: stronger continuous alert pattern

## 10. AI Concept Mapping Table
| AI Concept | How Implemented in Project |
|---|---|
| Rule-Based System | R1-R4 conditions based on closure time and blink rate |
| Knowledge Base | Threshold-driven rule definitions in backend detector logic |
| Inference Engine | Sequential rule matching + final state inference |
| Forward Chaining | Inputs -> rule checks -> matched rules -> inferred state |
| State Space Search | Selection among ALERT/TIRED/DROWSY/CRITICAL |
| Heuristic | Weighted fatigue score from closure and blink metrics |
| Expert Decision Layer | State-to-action mapping for warnings/alarms |

## 11. What Was Changed in Code
Main modified files:
- [backend/src/drowsiness_detector.py](backend/src/drowsiness_detector.py)
- [backend/src/api_server.py](backend/src/api_server.py)
- [backend/src/main.py](backend/src/main.py)
- [frontend/app/index.html](frontend/app/index.html)
- [frontend/app/app.js](frontend/app/app.js)
- [frontend/app/style.css](frontend/app/style.css)

Important additions:
- New config fields for AI thresholds
- Blink tracking and blink-rate calculation
- Rule matching with matched_rules trace
- Heuristic score generation
- State search and action mapping
- Frontend controls and live AI metrics display

## 12. Runtime Outputs You Can Show Teacher
From live status panel/API, you can demonstrate:
- state
- decision_action
- closed_seconds
- blink_rate_per_min
- heuristic_score
- matched_rules

This proves the system is doing AI reasoning, not only raw CV detection.

## 13. Demo Script (How to Explain in Class)
Use this short explanation flow:

1. "First we detect eyes and compute EAR from camera frames."
2. "Then we derive AI facts: eye closure duration and blink rate."
3. "These facts are sent to a rule-based knowledge base."
4. "Forward chaining checks which rules match."
5. "From possible states, we use heuristic search to choose best state."
6. "Finally action layer triggers warning or alarm based on severity."

## 14. Sample Scenarios for Viva
Scenario A:
- closed_seconds = 1.2
- blink_rate = 14
- Expected state: ALERT
- Action: NO_ACTION

Scenario B:
- closed_seconds = 2.0
- blink_rate = 8
- Expected state: TIRED
- Action: DISPLAY_WARNING

Scenario C:
- closed_seconds = 4.1
- blink_rate = 9
- Expected state: DROWSY
- Action: SOUND_ALARM

Scenario D:
- closed_seconds = 6.3
- blink_rate = 6
- Expected state: CRITICAL
- Action: CONTINUOUS_ALERT

## 15. Why This Qualifies as AI Lab Project
Because the final output is produced through AI reasoning components:
- explicit knowledge representation (rules)
- inference mechanism (forward chaining)
- search over state space
- heuristic-guided decision
- explainable output (matched rules + score + state + action)

So this project is now a complete AI-assisted decision system built over vision inputs.

## 16. Common Viva Questions and Ready Answers
Q1. Is this machine learning?
- No, this version is rule-based AI (expert system), not trained ML.

Q2. What is intelligent part here?
- The inference engine that reasons over rules and selects the best state using heuristics.

Q3. Why not use only one threshold?
- Multiple rules model realistic fatigue progression and provide explainable decisions.

Q4. Where is search used?
- In selecting the best state from ALERT/TIRED/DROWSY/CRITICAL using heuristic closeness.

Q5. How is this explainable?
- We expose matched_rules, heuristic_score, and decision_action for each output.

## 17. Future Improvements
- Add fuzzy rules to handle uncertainty
- Add head pose and yawning as additional facts
- Learn thresholds per driver profile
- Add event logging and fatigue trend chart

## 18. One-Line Conclusion
The project is transformed from a simple vision detector into an explainable AI expert system for real-time driver fatigue reasoning.
