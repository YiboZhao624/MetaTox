# The following prompts contain the discriminatory content.


# reason prompt is good, so there is no difference from v1.
REASONPROMPTV2 = """<USER>: You are a helpful assistant aiming at hateful speech detection. For each message, give the reason why it is hateful. Here are 2 examples:
Example 1:
<USER>: Context 1: The Democrat party is like a dog sniffing its own tail. It stinks.
<ASSISTANT>: Analysis 1: The context is considered hateful because it uses a derogatory comparison to dehumanize a political group (the Democrat party) by likening it to a dog, which implies foolishness or absurdity. Additionally, the phrase 'it stinks' conveys a strong negative judgment, suggesting disdain and contempt. Such language can foster hostility and division, contributing to a culture of hate rather than constructive discourse.
Example 2:
<USER>: Context 2: The working middle class is an easy target! Have fun bleeding us dry with more taxes & Sisyphean social programs, elite upper class robber barons!
<ASSISTANT>: Analysis 2: The context is considered hateful because it employs inflammatory language by labeling the upper class as 'robber barons',which evokes historical connotations of greed and exploitation. This framing dehumanizes individuals within that socioeconomic group and fosters animosity. Additionally, the phrase 'bleeding us dry' expresses a strong sense of victimization and resentment, promoting division between classes rather than encouraging constructive dialogue. Such rhetoric can perpetuate class hatred and animosity.
<USER>: New Context: {context}
<ASSISTANT>: Analysis: """

# Add some predefined relations, but the relations are not limited to the predefined ones.
# Add the instruction that the entities should be the original words instead of the conclusion of the analysis.
TRIPLETPROMPTV2 = '''<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive an original context and an analysis. Your task is to extract triplets that trigger hate.
In general, the context is more important and the analysis is just for assisting. The output should start with 'Output' and be with the format (head object A, predicate, tail object B) like (King, is, man). Also, there is a special condition that if the two phrase or word co-appear, they will lead to hate, you can use the "co-appear" as the relation and the two phrases or words as the head object and the tail object. There are some recommanded relations, you can use them if they are suitable: is, demean, promote, refer to, imply, link, co-appear, insult, downplay, etc. For entities, it is better to use the original words instead of the conclusion of the analysis. Here are two examples:
Example 1:
<USER>: Input 1:  Context: the persecution of christians isn't as bad as the persecution of muslims.
Analysis: The context is considered hateful because it downplays the suffering of one group (Christians) by comparing it to the suffering of another (Muslims). 
This minimizes the severity of the persecution faced by Muslims and can be seen as a form of victim-blaming. 
Furthermore, the implication that Christians are being persecuted is not universally recognized or acknowledged, 
and this statement can be seen as perpetuating a narrative that is divisive and can be hurtful to those who experience persecution.
<ASSISTANT>: Output 1:
(persecution of Christians, is not as bad as, persecution of Muslims),
(comparing to suffering of Muslims, downplay, suffering of Christians)
Example 2:
<USER>: Input 2: Context: just shows how many retarded jerks we have living here buying everything he has. yea, that means your kids you stupid clueless parents. wise up.
Analysis:  The context is considered hateful because it uses derogatory language to insult individuals based on their socioeconomic status (\"retarded jerks\") and their ability to make informed decisions (\"clueless parents\"). 
The use of the word'retarded' is particularly egregious as it's a term that has been historically used to demean people with intellectual disabilities. 
This language is not only hurtful but also discriminatory, perpetuating stigma and hate towards those who may be vulnerable. 
The final sentence, \"wise up,\" is a condescending and aggressive statement that further escalates the hate speech.
<ASSISTANT>: Output 2:
(retarded jerk, co-appear, parents),
(retarded jerk, insult, individuals),
(wise up, condescend, aggressive statement)
<USER>: New Input: {context}
<ASSISTANT>: Output:'''


FILTERPROMPTV2 = """<USER>: Please select the triplet evoking the hate from the following triplets began with 'Input:' and within the []. In general, the triplet should be in form like (Object A, relation, Object B), where A is a noun phrase or word, relation is a predicate(verb or a linking verb).If the input triplets don't fit the format, you should rewrite the input triplets first.
The number of triplet is not fixed, can be one or several, you are required to select all hate triggered triplets. Finally, output the hate triggered triplets in [] only.
Here are two examples:
Example 1:
<USER>: Input 1: [(downplays suffering of Christians, by comparing to, suffering of Muslims),(persecution of Christians, is not as bad as, persecution of Muslims),(minimizes severity of persecution, faced by, Muslims),(implication that Christians are being persecuted, is not universally recognized, acknowledgment)]
<ASSISTANT>: Output 1: [(persecution of Christians, is not as bad as, persecution of Muslims),(comparing to suffering of Muslims, downplay, suffering of Christians),]
Example 2:
<USER>: Input 2: [(retarded jerks, shows, socioeconomic status),(clueless parents, buying everything, he has),(derogatory language, insult, individuals),(wise up, condescend, aggressive statement)]
<ASSISTANT>: Output 2: [(retarded jerk, insult, individuals), (wise up, condescend, aggressive statement)]
<USER>: Input: {context}
<ASSISTANT>: Output:"""

BASELINEPROMPT = """
<Instructions>: Only output a or b.

<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive a context. Your task is to judge whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should only output a or b without any other words.

CONTEXT: {context}

a: {{Hateful}}
b: {{Non-hatefule}}

<ASSISTANT>:Prediction: """

RAGBASELINEPROMPT = """
<Instructions>: Only output a or b.

<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive a context. Your task is to judge whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should only output a or b without any other words.

a: {{Hateful}}
b: {{Non-hatefule}}

Here are 2 examples:
Example 1:
<USER>: Context 1: {RAGcontext_1}
<ASSISTANT>: {RAGprediction_1}

Example 2:
<USER>: Context 2: {RAGcontext_2}
<ASSISTANT>: {RAGprediction_2}

New Input:
<USER>: Context: {context}
<ASSISTANT>:Prediction: """

GRAPHRAGPROMPTV2 = """
<Instructions>: Only output a or b.

<USER>: You are a helpful assistant skilled in inductive reasoning. For each message, you will receive a context and related triplets as reference. All the triplets will lead to hate. If the context contains the same meaning with one of the triplets, it is hateful. Otherwise, you should leverage your knowledge to judge it. Your task is to determine whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should only output a or b.

a: {{Hateful}}
b: {{Non-hatefule}}

CONTEXT: {context}
TRIPLETS: {triplets}

<ASSISTANT>:Prediction: """

GRAPH2SHOTSPROMPTV2 = """
<Instructions>: You are a helpful assistant skilled in inductive reasoning. For each message, you will receive a context and related triplets as reference. All the triplets will lead to hate. If the context contains the same meaning with one of the triplets, it is hateful. Otherwise, you should leverage your knowledge to judge it. Your task is to determine whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should only output a or b.

a: {{Hateful}}
b: {{Non-hatefule}}

Here are 2 examples:
Example 1:
<USER>: CONTEXT: omitting the part where you raped someone and threatened to kill yourself if they left you.
TRIPLETS: (bitch, degrade, someone), (someone, supporting, Muslims), (\"bitch\", co-appear, nigga), (nigga, co-appear, what), (nigga, co-appear, Muslims), (Muslims, threatens, user)
<ASSISTANT>: b

Example 2:
<USER>: CONTEXT: i know you muzzies are peaceful cause that what you keep telling us are not you late for blowing up a building or something.
TRIPLETS: (muzzies, dehumanizes, individuals), (muzzies, Promotes, hostility), (muzzies, is a derogatory slur, people of Middle Eastern descent), (individuals, dehumanizes, illegal immigrants), (illegal immigrants, co-appear, apartment building), (illegal immigrants, go back to your country, hostility), (illegal immigrants, preferred over, americans), (hostility, contributes, dismissiveness), (hostility, targeted, americans), (dismissiveness, implies, fabrication), (people of Middle Eastern descent, dehumanizes, dirty muzzies), (dirty muzzies, co-appear, americans)
<ASSISTANT>: a

New Input:
<USER>: CONTEXT: {context}
TRIPLETS: {triplets}
<ASSISTANT>: 
"""


GRAPHPROMPT_GPT = """You are an assistant skilled in inductive reasoning. For each message, you will receive a context and a set of related triplets as references. All the triplets will express hateful content. Your task is to determine whether the context is hateful or not, based on the triplets provided.

If the context expresses the same sentiment as one of the hateful triplets, select a: Hateful. If the context does not express hate, select b: Non-hateful.

TRIPLETS: {triplets}
CONTEXT: {context}

Answer choices:
a: Hateful
b: Non-hateful

Please respond with either a or b based on your judgment.

<ASSISTANT>:Prediction: """




REWRITEBASELINEPROMPT = """
<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive a context. Your task is to rewrite the context to make it not hateful but still convey the same meaning.
{context}
"""


