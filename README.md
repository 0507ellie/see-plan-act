# see-plan-act
 
A vision-language-action system where a robot perceives its environment through a camera, receives natural language commands, and uses an LLM to decompose tasks into executable primitives — with the ability to replan visually when things go wrong.
 
The core loop: **see → understand → plan → act → see again**
 
Built on [ManiSkill3](https://maniskill.readthedocs.io/en/latest/) for robot simulation, with a CLIP visual encoder, LLM planner (GPT-4o / Claude), and RL-trained policies for each subtask.
