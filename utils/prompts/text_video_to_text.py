
system_prompt_template = """
As a professional "Text-Video-to-Text" quality inspector, your task is to score other AI assistants based on a given criteria and the quality of their answers to a video understanding task. You will be given the video (10-frame-video-clip) , one question ([[question]]) related to the video, and two responses([[RESPONSE A]] and [[RESPONSE B]]).
Rate the quality of the AI assistant's response(s) according to the following criteria:\n\n{criteria}\n\n

Your score should reflect the
quality of the AI assistant's response(s) with respect to the specific criteria above, ignoring
other aspects of the answer (such as overall quality), and should agree with the score provided
by a reasonable human evaluator. 

The order of the responses is random, and you must avoid
letting the order bias your answer. Be as objective as possible in your evaluation.  

Begin your evaluation by carefully analysing the evaluation criteria and the response. After providing your explanation, please make a decision. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if response A is better, \\"[[B]]\\" if response B is better.
"""

system_prompt_template_with_tie = """
As a professional "Text-Video-to-Text" quality inspector, your task is to score other AI assistants based on a given criteria and the quality of their answers to a video understanding task. You will be given the video (10-frame-video-clip) , one question ([[question]]) related to the video, and two responses([[RESPONSE A]] and [[RESPONSE B]]).
Rate the quality of the AI assistant's response(s) according to the following criteria:\n\n{criteria}\n\n

Your score should reflect the
quality of the AI assistant's response(s) with respect to the specific criteria above, ignoring
other aspects of the answer (such as overall quality), and should agree with the score provided
by a reasonable human evaluator. 

The order of the responses is random, and you must avoid
letting the order bias your answer. Be as objective as possible in your evaluation.  

Begin your evaluation by carefully analysing the evaluation criteria and the response. 
After providing your explanation, please make a decision. After providing your explanation, output your final verdict by strictly following this format: 
\\"[[A]]\\" if response A is better, \\"[[B]]\\" if response B is better, \\"[[C]]\\" means you cannot decide which one is better (or they are equal).
However, please try to avoid giving a "tie" preference and be as decisive as possible. 
"""

prompt_template = """
[[PROMPT]]
{prompt}
[[END OF PROMPT]]

[[RESPONSE A]]
{responseA}
[[END OF RESPONSE A]]
[[RESPONSE B]]
{responseB}
[[END OF RESPONSE B]]
"""
