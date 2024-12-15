HF_TOKEN = "Replace with your HF token"

REASONPROMPT = """<USER>: You are a helpful assistant aiming at hateful speech detection. For each message, give the reason why it is hateful. Here are 2 examples:
Example 1:
<USER>: Context 1: The Democrat party is like a dog sniffing its own tail. It stinks.
<ASSISTANT>: Analysis 1: The context is considered hateful because it uses a derogatory comparison to dehumanize a political group (the Democrat party) by likening it to a dog, which implies foolishness or absurdity. Additionally, the phrase 'it stinks' conveys a strong negative judgment, suggesting disdain and contempt. Such language can foster hostility and division, contributing to a culture of hate rather than constructive discourse.
Example 2:
<USER>: Context 2: The working middle class is an easy target! Have fun bleeding us dry with more taxes & Sisyphean social programs, elite upper class robber barons!
<ASSISTANT>: Analysis 2: The context is considered hateful because it employs inflammatory language by labeling the upper class as 'robber barons',which evokes historical connotations of greed and exploitation. This framing dehumanizes individuals within that socioeconomic group and fosters animosity. Additionally, the phrase 'bleeding us dry' expresses a strong sense of victimization and resentment, promoting division between classes rather than encouraging constructive dialogue. Such rhetoric can perpetuate class hatred and animosity.
<USER>: New Context: {context}
<ASSISTANT>: Analysis: """


TRIPLETPROMPT = '''<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive an original context and an analysis. Your task is to extract triplets that trigger hate.
In general, the context is more important and the analysis is just for assisting. The output should start with 'Output' and be with the format (head object A, predicate, tail object B) like (King, is, man). Also, there is a special condition that if the two phrase or word co-appear, they will lead to hate, you can use the "co-appear" as the relation and the two phrases or words as the head object and the tail object. Here are two examples:
Example 1:
<USER>: Input 1:  Context: the persecution of christians isn't as bad as the persecution of muslims.
Analysis: The context is considered hateful because it downplays the suffering of one group (Christians) by comparing it to the suffering of another (Muslims). 
This minimizes the severity of the persecution faced by Muslims and can be seen as a form of victim-blaming. 
Furthermore, the implication that Christians are being persecuted is not universally recognized or acknowledged, 
and this statement can be seen as perpetuating a narrative that is divisive and can be hurtful to those who experience persecution.
<ASSISTANT>: Output 1:
(persecution of Christians, is not as bad as, persecution of Muslims),
(comparing to suffering of Muslims, downplays, suffering of Christians)
Example 2:
<USER>: Input 2: Context: just shows how many retarded jerks we have living here buying everything he has. yea, that means your kids you stupid clueless parents. wise up.
Analysis:  The context is considered hateful because it uses derogatory language to insult individuals based on their socioeconomic status (\"retarded jerks\") and their ability to make informed decisions (\"clueless parents\"). 
The use of the word'retarded' is particularly egregious as it's a term that has been historically used to demean people with intellectual disabilities. 
This language is not only hurtful but also discriminatory, perpetuating stigma and hate towards those who may be vulnerable. 
The final sentence, \"wise up,\" is a condescending and aggressive statement that further escalates the hate speech.
<ASSISTANT>: Output 2:
(retarded, co-appear, jerk),
(derogatory language, insults, individuals),
(wise up, is condescending, aggressive statement)
<USER>: New Input: {context}
<ASSISTANT>: Output:'''


FILTERPROMPT = """<USER>: Please select the triplet evoking the hate from the following triplets began with 'Input:' and within the []. In general, the triplet should be in form like (Object A, relation, Object B), where A is a noun phrase or word, relation is a predicate(verb or a linking verb).If the input triplets don't fit the format, you should rewrite the input triplets first.
The number of triplet is not fixed, can be one or several, you are required to select all hate triggered triplets. Finally, output the hate triggered triplets in [] only.
Here are two examples:
Example 1:
<USER>: Input 1: [(downplays suffering of Christians, by comparing to, suffering of Muslims),(persecution of Christians, is not as bad as, persecution of Muslims),(minimizes severity of persecution, faced by, Muslims),(implication that Christians are being persecuted, is not universally recognized, acknowledgment)]
<ASSISTANT>: Output 1: [(persecution of Christians, is not as bad as, persecution of Muslims),(comparing to suffering of Muslims, downplays, suffering of Christians),]
Example 2:
<USER>: Input 2: [(retarded jerks, shows, socioeconomic status),(clueless parents, buying everything, he has),(derogatory language, insults, individuals),(wise up, is condescending, aggressive statement)]
<ASSISTANT>: Output 2: [(derogatory language, insults, individuals)]
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

GRAPHRAGPROMPT = """
<Instructions>: Only output a or b.

<USER>: You are a helpful assistant skilled in inductive reasoning. For each message, you will receive a context and related triplets as reference. All the triplets will lead to hate. If the context contains the same meaning with one of the triplets, it is hateful. Otherwise, you should leverage your knowledge to judge it. Your task is to determine whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should only output a or b.

a: {{Hateful}}
b: {{Non-hatefule}}

TRIPLETS: {triplets}
CONTEXT: {context}

<ASSISTANT>:Prediction: """

GRAPH2SHOTSPROMPT = """
<Instructions>: You are a helpful assistant skilled in inductive reasoning. For each message, you will receive a context and related triplets as reference. All the triplets will lead to hate. If the context contains the same meaning with one of the triplets, it is hateful. Otherwise, it is more possible to be benign, but you should leverage your knowledge to judge it. Your task is to determine whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should only output a or b.

a: {{Hateful}}
b: {{Non-hatefule}}

Here are 2 examples:
Example 1:
<USER>: TRIPLETS: (niggers, is described as, stupid), 
CONTEXT: Niggers are stupid.
<ASSISTANT>: a

Example 2:
<USER>: TRIPLETS: (white woman, shows, racism), (racism, aiming at, black man)
CONTEXT: That old white lady condemns any type of racism against black men.
<ASSISTANT>: b

New Input:
<USER>: TRIPLETS: {triplets}
CONTEXT: {context}
<ASSISTANT>: """


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


