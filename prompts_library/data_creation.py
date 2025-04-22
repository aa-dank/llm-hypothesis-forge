## LLM Prompt for Assessing Research Paper Inclusion in the research dataset

neuroscience_inclusion_assessment_prompt = """
You are a research assistant responsible for deciding whether a given research paper should be included in our dataset. The dataset requires papers to meet specific criteria regarding licensing, domain relevance, study design, and clarity of results. Please analyze the abstract (and any additional metadata) according to the criteria below and provide a final decision. If the paper does not meet one or more criteria, state which ones are not met.

### 1. **Basic Metadata**  
- Does the paper have structured abstract summarizing key research findings?

### 2. **Domain Relevance**
   - The paper must fall within the relevant domain(s) of {{ domain }}.

### 3. **Scientific Rigor and Clarity**  
   - The paper must present a clearly described **research question**, **methods**, and **results**—enough so they could be turned into a test item.  
   - A coherent study design should yield at least one **central empirical finding** or key result.

### 4. **Modifiable Results Section**  
   - The abstract should include a clear results portion that can be **logically altered** (e.g., reversing the direction of the outcome) while keeping methods/background intact.  
   - Very brief abstracts or ones lacking explicit results are not suitable.

### 5. **Non-Trivial Results**  
   - The main findings should require **domain-specific insight** to interpret. Trivial or purely descriptive findings (e.g., "We found no difference at all" with no meaningful context) are less suitable.

### 6. **Copyright or Memorization Concerns** *(if relevant)*  
   - Verify that the text is not **verbatim** from well-known sources that a large language model might have memorized (e.g., famous speeches, heavily cited reviews).  
   - If the text is suspected to be in the model's training set in a memorized form, consider excluding it.

---

### **Format Your Response as JSON**
Please format your response as a valid JSON object with the following structure:

```json
{
  "decision": "Include" or "Exclude", 
  "criteria_assessments": {
    "basic_metadata": {"met": true/false, "explanation": "..."},
    "domain_relevance": {"met": true/false, "explanation": "..."},
    "scientific_rigor": {"met": true/false, "explanation": "..."},
    "modifiable_results": {"met": true/false, "explanation": "..."},
    "non_trivial_results": {"met": true/false, "explanation": "..."},
    "copyright_concerns": {"met": true/false, "explanation": "..."}
  },
  "explanation": "Overall explanation of the decision"
}
```

---

### **Abstract to Analyze**:
{% if title %}**Title**: {{ title }}{% endif %}
{% if authors %}**Authors**: {{ authors }}{% endif %}
**Abstract**:
{{ abstract }}
"""

neuroscience_abstract_modification_prompt = """
Your task is to modify an abstract from a neuroscience research paper such that the changes significantly alter the result of the study without changing the methods and background. This way we can test the Artificial Intelligence understanding of the abstract’s subject area.
Please read the instructions below and ensure you follow them one by one while you are modifying the abstracts:
- The format to submit is putting double brackets around the change with the first element being the original and the second element being your edit. E.g., [[original passage, modified passage]]. Always remember to wrap your edits with the double brackets; there should not be any other edits outside the brackets to the original abstract.
- If you change a single word, never wrap the entire sentence inside the double brackets. For example, ‘... exhibit [[enhanced LTP and deficits in LTD, impaired LTP and enhanced LTD]].’ is a wrong format, the cor- rect format is: ‘... exhibit [[enhanced, impaired]] LTP and [[deficits, enhanced]] in LTD.’
- The beginning of an abstract is the background and methods, so you should not alter those parts of the abstract. Do not alter the first couple sentences.
- We want the abstract to become empirically wrong, but not logically incoherent.
- To find the original result of the paper, one should require some neuroscience insight, not just general reasoning ability. So it is critical that the changes you make don’t evaluate the Artificial Intelligence reasoning ability, but its knowledge of neuroscience and how the brain works.
- Watch out for making changes that alter the results, but may still have occurred in the authors’ study. For example, an fMRI abstract on learning might mention the hippocampus and not the striatum. Nevertheless, the striatum might have also been active and not reported in the abstract because it was not the focus of the study.
- The changes you make should not be identifiable or decodable from the rest of the abstract. Hence, if you make a change, make sure you change everything that can reveal the original abstract. For example, ‘activation of neurons in the visual cortex [[increases, decreases]] the activity in the motor cortex. This decrease in the activity of the visual cortex was followed by an increase in task performance.’. In this case it is very clear that the correct word is ‘decreases’ as the next sentence (‘This decrease in the activity of the visual cortex’) reveals that.
- Be mindful of the article when you change words. For example, if you change the word ‘decline’ to ‘enhancement’, you must change the article as well, so the change will be [[a decline, an enhancement]].
- Ensure that your edits maintain inter-sentence consistency and proper syntax. The changes should not contradict or confuse the overall meaning of the abstract.
- Avoid making trivial edits that do not require understanding of scientific concepts. The edits should reflect a deep understanding of the subject matter. - Do not miss any crucial results or findings in the abstract while making the edits. Every significant point should be addressed in your modifications.
To generate better responses, you can use the topic of their study and purpose of studies in those topics. This knowledge helps you to find what modification you should do in the abstract. Topics are:
- Behavioral/Cognitive: To understand how the brain influences behavior, cognition, and emotion, and to apply this understanding in diagnosing and treating neurological and psychiatric disorders.
- Cellular/Molecular: To study are to understand the functions and mechanisms of neurons at a cellular and molecular level, which includes investigating the biology of nerve cells, their genetic makeup, and how they form complex circuits, ultimately contributing to our understanding of brain function, behavior, and the development of treatments for neurological disorders.
- Neurobiology of Disease: To understand the biological basis of various neurological and psychiatric disorders in order to develop effective treatments and preventative measures.
- Development/Plasticity/Repair: to understand the mechanisms of brain development, adaptation, and repair in response to injury or disease, with the goal of developing strategies and treatments to enhance brain recovery and function.
- Systems/Circuits: to understand how neural circuits in the brain interact and coordinate to process information, control behavior, and support cognitive functions.
Here are two examples of the edited abstract by human experts which can help you to understand the task:
Example 1: {{ example_1 }}
Example 2: {{ example_2 }}
These are some common mistakes you have made in the past.
So keep them in mind whilst generating your responses:
- You misunderstood/ignore the information provided at the beginning of the abstract.
- The edits you have made are not what we are aiming for, you tweaked a portion of the studies with non-significant findings, so there’s no significant alternation of results occurring. Make sure your edit changes the main results of the studies, not trivial changes.
- Lack of inter-sentence consistency in the prompt
- You made edits as early as the first sentence. THe first few sentence are general knowledge and are not result of the study. So you shouldn’t make any change in the beginning.
- Most of your edits contradict the conclusion. Make sure your changes do not contradict the conclusions or any part of the abstract.
- You only modified verbs the understanding of which does not require understanding of scientific concepts & names of compounds, which makes the edits less likely to do wrong as long as reasons logically
- One of your edits contradicts all other edits.
- Your edit is inconsistent with the beginning of the sentence
- You failed to change the first part of the conclusion for consistency
- You missed out on one change.
- You misunderstood the purpose of the study. Although in the abstract it explicitly states the purpose of the study.
Below, you are given an abstract with its topic. Follow the instructions given to you and return the modified abstract. Remember to use double brackets to show the changes ([[original, modified]] and keep the rest of the abstract unchanged. Also, pay attention to all the information you were given above as well as the common mistakes you have made before.
Abstract to edit:
Topic: {{ domain }}
Abstract: {{ abstract_to_edit }}
"""

biotechnology_abstract_modification_prompt = """
Your task is to modify an abstract from a biotechnology research paper such that the changes significantly alter the result of the study without changing the methods and background. This way we can test the Artificial Intelligence's understanding of the abstract’s subject area.

Please follow these instructions carefully:
- Use double brackets to wrap each modification, with the first element being the original text and the second your edited text (e.g., [[original passage, modified passage]]). Ensure that no other parts of the abstract are modified outside these brackets.
- If you change a single word, modify only that word within the double brackets rather than the whole sentence.
- Do not alter the initial portion of the abstract which covers background and methodology (the first couple of sentences).
- The modified abstract must become empirically incorrect without becoming logically incoherent.
- Ensure that every change disguises clues to the original abstract; avoid edits that can be easily decoded.
- Mind punctuation and article consistency. For example, if you change 'a decrease' to 'an increase', adjust the article accordingly.
- Maintain inter-sentence consistency and proper syntax throughout your edits.
- Avoid trivial modifications; focus on alterations that affect key findings related to biotechnology topics such as molecular biology techniques, protein engineering, genetic modifications, or industrial bioprocess strategies.

Below are two examples of edited abstracts by human experts for guidance:
Example 1: {{ example_1 }}
Example 2: {{ example_2 }}

Avoid common pitfalls:
- Do not modify the introductory sentences outlining methods or background.
- Do not make superficial edits that do not alter critical experimental results.
- Ensure all edits are consistent and do not contradict each other.
- Always check that the modifications significantly affect the study’s outcomes without altering the methodological description.

Abstract to edit:
Topic: {{ domain }}
Abstract: {{ abstract_to_edit }}
"""

economics_inclusion_assessment_prompt = """
You are a research assistant responsible for deciding whether a given research paper should be included in our dataset. The dataset requires papers to meet specific criteria regarding licensing, domain relevance, study design, and clarity of results. Please analyze the abstract (and any additional metadata) according to the criteria below and provide a final decision. If the paper does not meet one or more criteria, state which ones are not met.

### 1. **Basic Metadata**  
   - Does the paper have a structured abstract summarizing key research findings?

### 2. **Domain Relevance**
   - The paper must fall within the relevant domain(s) of economics, including but not limited to applied microeconomics, macroeconomics, development, labor, policy evaluation, econometrics, and finance.

### 3. **Scientific Rigor and Clarity**  
   - The paper must present a clearly described **research question**, **methods**, and **results**—enough so they could be turned into a test item.  
   - A coherent study design should yield at least one **central empirical finding** or key result.

### 4. **Modifiable Results Section**  
   - The abstract should include a clear results portion that can be **logically altered** (e.g., reversing the sign or direction of an effect, changing the conclusion about significance) while keeping methods/background intact.  
   - Very brief abstracts or ones lacking explicit results are not suitable.

### 5. **Non-Trivial Results**  
   - The main findings should require **domain-specific insight** to interpret. Trivial or purely descriptive findings (e.g., “We summarize prior work on inflation trends”) are less suitable.

### 6. **Copyright or Memorization Concerns** *(if relevant)*  
   - Verify that the text is not **verbatim** from well-known sources that a large language model might have memorized (e.g., textbook definitions, widely cited canonical abstracts).  
   - If the text is suspected to be in the model’s training set in a memorized form, consider excluding it.

---

### **Format Your Response as JSON**
Please format your response as a valid JSON object with the following structure:

```json
{
  "decision": "Include" or "Exclude", 
  "criteria_assessments": {
    "basic_metadata": {"met": true/false, "explanation": "..."},
    "domain_relevance": {"met": true/false, "explanation": "..."},
    "scientific_rigor": {"met": true/false, "explanation": "..."},
    "modifiable_results": {"met": true/false, "explanation": "..."},
    "non_trivial_results": {"met": true/false, "explanation": "..."},
    "copyright_concerns": {"met": true/false, "explanation": "..."}
  },
  "explanation": "Overall explanation of the decision"
}
```

---

### **Abstract to Analyze**:
{% if title %}**Title**: {{ title }}{% endif %}
{% if authors %}**Authors**: {{ authors }}{% endif %}
**Abstract**:
{{ abstract }}
"""

economics_abstract_modification_prompt = """
Your task is to modify an abstract from an economics research paper such that the changes significantly alter the result of the study without changing the methods and background. This way we can test the Artificial Intelligence understanding of the abstract’s subject area.
Please read the instructions below and ensure you follow them one by one while you are modifying the abstracts:
- The format to submit is putting double brackets around the change with the first element being the original and the second element being your edit. E.g., [[original passage, modified passage]]. Always remember to wrap your edits with the double brackets; there should not be any other edits outside the brackets in the original abstract.
- If you change a single word, never wrap the entire sentence inside the double brackets. For example, “... exhibit [[increased, decreased]] economic growth and [[reduced, increased]] inflation.” is the correct format (only change the key words, not the whole sentence).
- The beginning of an abstract is the background and methods, so you should not alter those parts. Do not alter the first couple of sentences.
- We want the abstract to become empirically incorrect (or inconsistent with the actual findings), but not logically incoherent.
- To identify the original result of the paper, one should require some economics insight, not just general reasoning ability. It is critical that the changes you make test the model's understanding of economic principles and how economic systems operate, rather than general logic.
- Watch out for making changes that alter the results in a way that might still be plausible in an economic study. For example, an abstract on the impact of a tax policy might report changes in consumption behavior; ensure your modifications render the results significantly different from the original findings.
- The changes you make should not be identifiable or decodable from the rest of the abstract. Hence, if you make a change, make sure you change all parts that could reveal the original result. For example, “an increase in consumer spending [[increases, decreases]] GDP growth. This decrease in consumer spending was followed by a reduction in economic output.” In this case, the modification should be comprehensive enough to obscure the original finding.
- Be mindful of grammatical consistency when you change words. For example, if you change “a decline” to “an enhancement”, ensure that the article is changed as well: [[a decline, an enhancement]].
- Ensure that your edits maintain inter-sentence consistency and proper syntax. The changes should not contradict or confuse the overall meaning of the abstract.
- Avoid making trivial edits that do not require understanding of economic concepts. The edits should reflect a deep understanding of the subject matter. Do not miss any crucial results or findings in the abstract; every significant point should be addressed in your modifications.

To generate better responses, you can use the topic of the study and its economic focus to guide your modifications. Topics are, for example:
- Behavioral/Experimental Economics: To understand how individual decision-making and behavior are influenced by psychological, social, and cognitive factors, and how these affect economic outcomes.
- Econometrics/Quantitative Analysis: To analyze economic data using statistical methods, ensuring robust estimation of relationships and effects.
- Macroeconomics: To study aggregate economic phenomena such as growth, inflation, unemployment, and the effects of monetary and fiscal policy.
- Microeconomics: To examine the behavior of individual agents and firms, market mechanisms, and resource allocation.
- Development Economics: To investigate determinants of economic growth and development in emerging economies, including issues of inequality, poverty, and institutional change.
- Financial Economics: To study the dynamics of financial markets, asset pricing, and risk management.

Here are two examples of the edited abstract by human experts which can help you understand the task:
Example 1: {{ example_1 }}
Example 2: {{ example_2 }}

These are some common mistakes you have made in the past:
- You misunderstood or ignored the information provided at the beginning of the abstract.
- The edits you made are not significant enough; you tweaked a portion of the study with non-significant findings, so there’s no substantial alteration of the main results. Make sure your edit changes the main results of the study, not trivial parts.
- Lack of inter-sentence consistency.
- You made edits as early as the first sentence. The first few sentences are background and methods and should not be altered.
- Most of your edits contradict the conclusion. Ensure your changes do not contradict the conclusions or any part of the abstract.
- You only modified verbs without addressing the deeper economic implications, making the edits less significant.
- One of your edits contradicts all other edits.
- Your edit is inconsistent with the beginning of the sentence.
- You failed to change the first part of the conclusion for consistency.
- You missed out on one change.
- You misunderstood the purpose of the study, even if the abstract explicitly states it.

Below, you are given an abstract with its topic. Follow the instructions given to you and return the modified abstract. Remember to use double brackets to show the changes ([[original, modified]]) and keep the rest of the abstract unchanged. Also, pay attention to all the information provided above as well as the common mistakes you have made before.

Abstract to edit: Topic: {{ domain }}
Abstract: {{ abstract_to_edit }}
"""

abstract_edit_verification_prompt = """
## Abstract Modification Verification

Now that you've modified the abstract, carefully verify your work to ensure you haven't made any of these common errors:

### Self-Check Checklist:

1. **Background vs. Results**: 
   - Did you preserve the first few sentences (background/methods) without modifications?
   - Did you only modify the results section, not the introduction or methodology?

2. **Significance of Changes**:
   - Are your modifications substantial enough to meaningfully alter the main economic findings?
   - Did you avoid making trivial edits that don't significantly change the core results?

3. **Consistency Checks**:
   - Are all your edits consistent with each other?
   - Do your edits maintain logical flow between sentences?
   - Have you ensured there are no contradictions between different parts of the modified abstract?
   - Are your modifications consistent with the conclusion or final statements?
   - If you changed part of a conclusion, did you appropriately modify all related statements?

4. **Technical Depth**:
   - Did you modify domain-specific economic concepts rather than just general verbs or descriptions?
   - Do your edits require subject matter expertise in economics to identify as incorrect?

5. **Format Compliance**:
   - Have you used the correct [[original, modified]] format for all changes?
   - Did you avoid wrapping entire sentences when only changing specific terms?
   - Did you adjust articles (a/an) appropriately when making word changes?

6. **Contextual Understanding**:
   - Have you correctly understood the purpose of the study?
   - Are your modifications aligned with what the study was trying to investigate?

Please review your modifications against each point on this checklist. If any issues are found, revise your edits accordingly before submitting your final response.

{{ modified_abstract }}
"""

neuroscience_abstract_edit_verification_prompt = """
## Abstract Modification Verification - Neuroscience

Now that you've modified the abstract, carefully verify your work to ensure you haven't made any of these common errors:

### Self-Check Checklist:

1. **Background vs. Results**: 
   - Did you preserve the first few sentences (background/methods) without modifications?
   - Did you only modify the results section, not the introduction or methodology?

2. **Significance of Changes**:
   - Are your modifications substantial enough to meaningfully alter the main neuroscience findings?
   - Did you avoid making trivial edits that don't significantly change the core results?

3. **Consistency Checks**:
   - Are all your edits consistent with each other?
   - Do your edits maintain logical flow between sentences?
   - Have you ensured there is no contradiction between different parts of the modified abstract?
   - Are your modifications consistent with the conclusion/final statements?
   - If you changed part of a conclusion, did you appropriately modify all related statements?

4. **Technical Depth**:
   - Did you modify domain-specific neuroscience concepts (e.g., neural activation patterns, imaging metrics, synaptic changes) rather than just general verbs or descriptions?
   - Do your edits require neuroscience expertise to identify as incorrect?

5. **Format Compliance**:
   - Have you used the correct [[original, modified]] format for all changes?
   - Did you avoid wrapping entire sentences when only changing specific terms?
   - Did you adjust articles (a/an) appropriately when making word changes?

6. **Contextual Understanding**:
   - Have you correctly understood the purpose of the study?
   - Are your modifications aligned with what the study was trying to investigate?

Please review your modifications against each point on this checklist. If any issues are found, revise your edits accordingly before submitting your final response.

{{ modified_abstract }}
"""

economics_abstract_edit_verification_prompt = """
## Abstract Modification Verification - Economics

Now that you've modified the abstract, carefully verify your work to ensure you haven't made any of these common errors:

### Self-Check Checklist:

1. **Background vs. Results**: 
   - Did you preserve the initial sentences (background/methods) without modifications?
   - Did you only modify the results section, without altering the introduction or methodology?

2. **Significance of Changes**:
   - Are your modifications substantial enough to meaningfully alter the main economic findings?
   - Did you avoid making trivial edits that don't significantly change the core results?

3. **Consistency Checks**:
   - Are all your edits consistent with one another?
   - Do your edits maintain logical flow between sentences?
   - Have you ensured that there are no contradictions between different parts of the modified abstract?
   - Are your modifications consistent with the conclusion or final statements?
   - If you modified part of the conclusion, did you adjust all related statements accordingly?

4. **Technical Depth**:
   - Did you modify domain-specific economic concepts (e.g., econometric estimates, policy impact measures, growth rates, market behavior) rather than just general verbs or descriptions?
   - Do your edits require economic expertise to detect inaccuracies?

5. **Format Compliance**:
   - Have you used the correct [[original, modified]] format for all changes?
   - Did you avoid wrapping entire sentences when only specific terms need modification?
   - Did you adjust articles (a/an) appropriately when making word changes?

6. **Contextual Understanding**:
   - Have you correctly understood the purpose of the study?
   - Are your modifications aligned with the economic questions or hypotheses that the study intended to investigate?

Please review your modifications against each point on this checklist. If any issues are found, revise your edits accordingly before submitting your final response.

{{ modified_abstract }}
"""

optometry_inclusion_assessment_prompt = """
You are a research assistant responsible for deciding whether a given research paper should be included in our dataset. The dataset requires papers to meet specific criteria regarding licensing, domain relevance, study design, and clarity of results. Please analyze the abstract (and any additional metadata) according to the criteria below and provide a final decision. If the paper does not meet one or more criteria, state which ones are not met.

### 1. **Basic Metadata**  
- Does the paper have structured abstract summarizing key research findings?

### 2. **Domain Relevance**
   - The paper must fall within the relevant domain(s) of {{ domain }}.

### 3. **Scientific Rigor and Clarity**  
   - The paper must present a clearly described **research question**, **methods**, and **results**—enough so they could be turned into a test item.  
   - A coherent study design should yield at least one **central empirical finding** or key result.

### 4. **Modifiable Results Section**  
   - The abstract should include a clear results portion that can be **logically altered** (e.g., reversing the direction of the outcome) while keeping methods/background intact.  
   - Very brief abstracts or ones lacking explicit results are not suitable.

### 5. **Non-Trivial Results**  
   - The main findings should require **domain-specific insight** to interpret. Trivial or purely descriptive findings (e.g., "We found no difference at all" with no meaningful context) are less suitable.

### 6. **Copyright or Memorization Concerns** *(if relevant)*  
   - Verify that the text is not **verbatim** from well-known sources that a large language model might have memorized (e.g., famous speeches, heavily cited reviews).  
   - If the text is suspected to be in the model's training set in a memorized form, consider excluding it.

---

### **Format Your Response as JSON**
Please format your response as a valid JSON object with the following structure:

```json
{
  "decision": "Include" or "Exclude", 
  "criteria_assessments": {
    "basic_metadata": {"met": true/false, "explanation": "..."},
    "domain_relevance": {"met": true/false, "explanation": "..."},
    "scientific_rigor": {"met": true/false, "explanation": "..."},
    "modifiable_results": {"met": true/false, "explanation": "..."},
    "non_trivial_results": {"met": true/false, "explanation": "..."},
    "copyright_concerns": {"met": true/false, "explanation": "..."}
  },
  "explanation": "Overall explanation of the decision"
}
```

---

### **Abstract to Analyze**:
{% if title %}**Title**: {{ title }}{% endif %}
{% if authors %}**Authors**: {{ authors }}{% endif %}
**Abstract**:
{{ abstract }}
"""

optometry_abstract_modification_prompt = """
Your task is to modify an abstract from an optometry research paper such that the changes significantly alter the result of the study without changing the methods and background. This way we can test the Artificial Intelligence understanding of the abstract’s subject area.
Please read the instructions below and ensure you follow them one by one while you are modifying the abstracts:
- The format to submit is putting double brackets around the change with the first element being the original and the second element being your edit. E.g., [[original passage, modified passage]]. Always remember to wrap your edits with the double brackets; there should not be any other edits outside the brackets to the original abstract.
- If you change a single word, never wrap the entire sentence inside the double brackets. For example, ‘... exhibit [[enhanced LTP and deficits in LTD, impaired LTP and enhanced LTD]].’ is a wrong format, the correct format is: ‘... exhibit [[enhanced, impaired]] LTP and [[deficits, enhanced]] in LTD.’
- The beginning of an abstract is the background and methods, so you should not alter those parts of the abstract. Do not alter the first couple sentences.
- We want the abstract to become empirically wrong, but not logically incoherent.
- To find the original result of the paper, one should require some optometry insight, not just general reasoning ability. So it is critical that the changes you make don’t evaluate the Artificial Intelligence reasoning ability, but its knowledge of optometry and how the eyes works.
- Watch out for making changes that alter the results, but may still have occurred in the authors’ study. For example, an abstract on visual acuity testing might mention improvements in the central visual field and not the peripheral field. Nevertheless, the peripheral field might have also improved and not been reported in the abstract because it was not the focus of the study.
- The changes you make should not be identifiable or decodable from the rest of the abstract. Hence, if you make a change, make sure you change everything that can reveal the original abstract. For example, ‘exposure to bright light [[improves, worsens]] contrast sensitivity in patients with glaucoma. This improvement in contrast sensitivity was associated with better reading speed.’ In this case, it is very clear that the correct word is ‘improves’ as the next sentence (‘This improvement in contrast sensitivity’) reveals that.
- Be mindful of the article when you change words. For example, if you change the word ‘decline’ to ‘enhancement’, you must change the article as well, so the change will be [[a decline, an enhancement]].
- Ensure that your edits maintain inter-sentence consistency and proper syntax. The changes should not contradict or confuse the overall meaning of the abstract.
- Avoid making trivial edits that do not require understanding of scientific concepts. The edits should reflect a deep understanding of the subject matter. 
- Do not miss any crucial results or findings in the abstract while making the edits. Every significant point should be addressed in your modifications.
- To generate better responses, you can use the topic of their study and the purpose of studies in those topics. This knowledge helps you to find what modification you should do in the abstract. Topics are:
- Visual Perception/Behavioral: To understand how visual processing influences behavior, perception, and decision-making, and to apply this knowledge in diagnosing and managing vision-related behavioral issues, such as visual attention deficits or perceptual anomalies.
- Cellular/Molecular Vision Science: To study the functions and mechanisms of retinal and optic nerve cells at a cellular and molecular level, including phototransduction, gene expression in retinal cells, and the role of proteins in visual signal transmission, with the aim of understanding and treating retinal diseases.
- Ocular Disease Mechanisms: To understand the biological basis of common eye diseases such as glaucoma, macular degeneration, or diabetic retinopathy, in order to inform the development of effective diagnostic tools, treatments, and preventative strategies.
- Development/Plasticity/Repair in Vision: To understand how the visual system develops and adapts over time, particularly in response to injury, disease, or visual training, with the goal of promoting visual recovery and rehabilitation.
- Visual Systems and Neural Circuits: To study how different components of the visual system—from the retina to the visual cortex—work together to process visual information and support functions like depth perception, motion detection, and visual acuity.
Here are two examples of the edited abstract by human experts which can help you to understand the task:
Example 1: {{ example_1 }}
Example 2: {{ example_2 }}
These are some common mistakes you have made in the past.
So keep them in mind whilst generating your responses:
- You misunderstood/ignore the information provided at the beginning of the abstract.
- The edits you have made are not what we are aiming for, you tweaked a portion of the studies with non-significant findings, so there’s no significant alternation of results occurring. Make sure your edit changes the main results of the studies, not trivial changes.
- Lack of inter-sentence consistency in the prompt
- You made edits as early as the first sentence. The first few sentence are general knowledge and are not result of the study. So you shouldn’t make any change in the beginning.
- Most of your edits contradict the conclusion. Make sure your changes do not contradict the conclusions or any part of the abstract.
- You only modified verbs the understanding of which does not require understanding of scientific concepts & names of compounds, which makes the edits less likely to do wrong as long as reasons logically
- One of your edits contradicts all other edits.
- Your edit is inconsistent with the beginning of the sentence
- You failed to change the first part of the conclusion for consistency
- You missed out on one change.
- You misunderstood the purpose of the study. Although in the abstract it explicitly states the purpose of the study.
Below, you are given an abstract with its topic. Follow the instructions given to you and return the modified abstract. Remember to use double brackets to show the changes ([[original, modified]] and keep the rest of the abstract unchanged. Also, pay attention to all the information you were given above as well as the common mistakes you have made before.
Abstract to edit:
Topic: {{ domain }}
Abstract: {{ abstract_to_edit }}
"""

optometry_abstract_edit_verification_prompt = """
## Abstract Modification Verification - Optometry

Now that you've modified the abstract, carefully verify your work to ensure you haven't made any of these common errors:

### Self-Check Checklist:

1. **Background vs. Results**: 
   - Did you preserve the initial sentences (background/methods) without modifications?
   - Did you only modify the results section, without altering the introduction or methodology?

2. **Significance of Changes**:
   - Are your modifications substantial enough to meaningfully alter the main optometry findings?
   - Did you avoid making trivial edits that don't significantly change the core results?

3. **Consistency Checks**:
   - Are all your edits consistent with one another?
   - Do your edits maintain logical flow between sentences?
   - Have you ensured that there are no contradictions between different parts of the modified abstract?
   - Are your modifications consistent with the conclusion or final statements?
   - If you modified part of the conclusion, did you adjust all related statements accordingly?

4. **Technical Depth**:
   - Did you modify domain-specific optometric concepts (e.g., refractive error measurements, visual acuity scores, intraocular pressure levels, retinal imaging findings) rather than just general verbs or descriptions?
   - Do your edits require optometric expertise to detect inaccuracies?

   5. **Format Compliance**:
   - Have you used the correct [[original, modified]] format for all changes?
   - Did you avoid wrapping entire sentences when only specific terms need modification?
   - Did you adjust articles (a/an) appropriately when making word changes?

6. **Contextual Understanding**:
   - Have you correctly understood the purpose of the study?
   - Are your modifications aligned with the optometry questions or hypotheses that the study intended to investigate?

Please review your modifications against each point on this checklist. If any issues are found, revise your edits accordingly before submitting your final response.

{{ modified_abstract }}
"""

discern_abstracts_pair_prompt = """
You are given two abstracts: abstract_1 and abstract_2. Your task is to determine which of these two abstracts is more likely to be from a real research paper and which one is more likely to be fake. Please consider the true meaning of the abstracts and evaluate which one contains the most scientifically accurate information, logical coherence, and typical language found in genuine academic papers.

1. **Decision**: Choose either 1 or 2 to indicate which abstract is the real one. You must choose one and only one, and provide no other option.
2. **Explanation**: Provide a 5-sentence explanation for your decision. Your explanation should focus on:
   - The logical structure and coherence of the content.
   - The appropriateness of the language used in each abstract.
   - The presence of scientific or academic tone and references that are typically found in real papers.
   - Any signs that suggest one abstract might be fabricated or too vague, nonspecific, or overly simplistic to be from a legitimate academic paper.
   - Identify if either abstract contains incorrect information or statements that are misleading, inaccurate, or factually false. If one abstract contains false information, this should strongly influence your decision.

Your output should be a JSON object with the following keys:
'''
{{
    "decision": <1 or 2>,
    "explanation": "<5-sentence explanation>"
}}
'''

Your output should be nothing but that json file.

abstract_1 = "{abstract_1}"
abstract_2 = "{abstract_2}"
"""
