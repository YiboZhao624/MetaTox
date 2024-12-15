REASONPROMPTV3 = """<USER>: You are a professional content moderator specializing in hate speech detection. Your task is to analyze whether a given message contains hate speech and explain the specific reasons why it is considered hateful. Please focus on identifying key hateful elements like derogatory language, dehumanization, discrimination, or inflammatory rhetoric. Here are 2 examples:

Example 1:
<USER>: Context 1: The Democrat party is like a dog sniffing its own tail. It stinks.
<ASSISTANT>: Analysis 1: This message contains hate speech for two key reasons:
1. It uses dehumanizing language by comparing a political party to a dog in a demeaning way
2. The phrase "it stinks" expresses explicit contempt and hostility rather than legitimate criticism
These elements combine to promote hatred and division rather than constructive political discourse.

Example 2:
<USER>: Context 2: The working middle class is an easy target! Have fun bleeding us dry with more taxes & Sisyphean social programs, elite upper class robber barons!
<ASSISTANT>: Analysis 2: This message contains hate speech for these specific reasons:
1. It uses the inflammatory term "robber barons" to demonize the upper class
2. The phrase "bleeding us dry" promotes a victim narrative and class resentment
3. The overall tone encourages hostility between social classes rather than understanding
These elements work together to incite class-based hatred and division.

<USER>: New Context: {context}
<ASSISTANT>: Analysis: Let me identify the specific hateful elements in this message:"""


TRIPLETPROMPTV3 = '''<USER>: You are a helpful AI assistant specialized in extracting relationship triplets from text and its analysis. Your task is to identify triplets that indicate hate or bias, using both the original context and its analysis.

Instructions:
1. Output Format: (head, relation, tail)
2. Key Relations:
   - is
   - demean
   - promote
   - refer to
   - imply
   - link
   - co-appear (use when two phrases together trigger hate)
   - insult
   - downplay

3. Important Guidelines:
   - Focus on the context first, use analysis as supporting information
   - Use original words from the text when possible
   - Extract only hate-indicating relationships

Examples:

Example 1:
Context: "the persecution of christians isn't as bad as the persecution of muslims."
Analysis: The context is considered hateful because it downplays the suffering of one group (Christians) by comparing it to the suffering of another (Muslims). This minimizes the severity of the persecution faced by Muslims and can be seen as a form of victim-blaming.
Output:
(persecution of Christians, is not as bad as, persecution of Muslims)
(comparing to suffering of Muslims, downplay, suffering of Christians)

Example 2:
Context: "just shows how many retarded jerks we have living here buying everything he has. yea, that means your kids you stupid clueless parents. wise up."
Analysis: The context is considered hateful because it uses derogatory language to insult individuals and their ability to make informed decisions. The use of the word 'retarded' is particularly offensive as it demeans people with disabilities. The tone is condescending and aggressive.
Output:
(retarded jerk, co-appear, parents)
(retarded jerk, insult, individuals)
(wise up, condescend, aggressive statement)

Now, please analyze this input:
Context: {context}
Analysis: {analysis}

<ASSISTANT>: Output:'''


FILTERPROMPTV3 = '''<USER>: You are a helpful AI assistant specialized in identifying hate-triggering relationships from triplets. Your task is to analyze the given triplets and select only those that directly evoke hate or bias.

Rules for Selection:
1. Triplet Format: (Subject, Relation, Object)
   - Subject: Usually a noun phrase
   - Relation: A verb or linking phrase
   - Object: The target of the relation

2. Selection Criteria:
   - Select triplets that directly indicate hate or bias
   - Include both explicit and implicit hate indicators
   - Maintain the original format of valid triplets
   - Rewrite malformed triplets to match the format

Here are two examples to guide you:

Example 1:
Input: [
    (downplays suffering of Christians, by comparing to, suffering of Muslims),
    (persecution of Christians, is not as bad as, persecution of Muslims),
    (minimizes severity of persecution, faced by, Muslims),
    (implication that Christians are being persecuted, is not universally recognized, acknowledgment)
]
Output: [
    (persecution of Christians, is not as bad as, persecution of Muslims),
    (comparing to suffering of Muslims, downplay, suffering of Christians)
]

Example 2:
Input: [
    (retarded jerks, shows, socioeconomic status),
    (clueless parents, buying everything, he has),
    (derogatory language, insult, individuals),
    (wise up, condescend, aggressive statement)
]
Output: [
    (retarded jerk, insult, individuals),
    (wise up, condescend, aggressive statement)
]

Now, please analyze these triplets and select those that evoke hate:
Input: {context}

<ASSISTANT>: Output:'''