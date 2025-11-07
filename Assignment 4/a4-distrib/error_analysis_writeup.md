# Part 3: Error Analysis - EntailmentFactChecker

## Error Categories

After looking through my false positives and false negatives, I identified a few key patterns in where my model went wrong. I examined all 8 false positives and 10 false negatives my system produced.

### False Positives: When my model was too confident

**Category 1: Getting tricked by similar terms**
My model kept accepting facts when it saw related words, even when they weren't quite right. For example, it thought "engraver" was close enough to "printmaker" to call it supported. Same thing happened with "military sniper" vs "served as a sniper" - they sound similar, but one talks about his profession while the other talks about what he did. This explains why I had trouble lowering my threshold below 0.80 during tuning - going lower created too many of these mistakes.

**Category 2: Making assumptions from context**
The model sometimes saw relevant information and just assumed the specific claim must be true. Like when it saw Travis Oliphant had different roles at different places and concluded he "worked as a researcher at various organizations" - but the passage never actually says that. Or calling Mayo Clinic an "organization" just because it's mentioned as a school. Technically true maybe, but not what the passage states.

### False Negatives: When my model was too picky

**Category 3: Missing obvious paraphrases**
Some facts were clearly supported, just not in the exact words. The passage said Gerhard Fischer "contributed to the development and popularity of the hand held metal detector" and I asked if he was "best known for inventing a metal detector" - my model said no. That's being way too strict. These were really frustrating because they seemed so obvious when I reviewed them.

**Category 4: Not catching embedded details**
This was my biggest problem (7 out of 10 false negatives!). When a passage said "1995's Just Cause", my model didn't recognize that supports BOTH "appeared in Just Cause" AND "Just Cause was released in 1995". The information was there, just tucked inside a larger sentence. Same with "She appeared in the movie 'Belly' in 1998" - literally has both facts right there, but my model rejected both.

This pattern explains why I got stuck at 81.4% accuracy. I needed a lower threshold to catch these obvious cases, but that would create more false positives from Category 1.

## Statistics

- **False Positives:** 8 total
  - Similar terms: 4 cases
  - Context overgeneralization: 4 cases

- **False Negatives:** 10 total
  - Paraphrasing: 3 cases
  - Embedded info: 7 cases

## Three Detailed Examples

### Example 1: Jean Daullé the "printmaker" (False Positive)

**Fact:** Jean Daullé was a printmaker.

**Ground Truth:** NS

**My Prediction:** S

**Category:** Similar terms confusion

**Why I got it wrong:**
The passage says Jean Daullé was a "French engraver." I'm guessing my entailment model saw "engraver" and "printmaker" as close enough since they're both artistic professions involving making images. But actually, engraving is a specific technique - you cut into metal plates. Printmaking is broader and includes lots of other methods. So technically, all engravers might be printmakers in a general sense, but the passage specifically calls him an engraver, not a printmaker. My model should have caught that distinction.

### Example 2: "She appeared in Just Cause" (False Negative)

**Fact:** She appeared in Just Cause.

**Ground Truth:** S

**My Prediction:** NS

**Category:** Embedded information

**Why I got it wrong:**
This one frustrated me when I reviewed it. The passage literally says "1995's 'Just Cause' with Sean Connery and Laurence Fishburne" when listing Taral Hicks' film roles. It's RIGHT THERE. My model must have gotten confused by the formatting or the way it was embedded in the list. Maybe the entailment score was just barely below my threshold (0.81), so it rejected it. Looking back, this is exactly the kind of obvious case I was trying to catch by testing lower thresholds during tuning, but going lower introduced other errors.

### Example 3: Travis Oliphant the "researcher" (False Positive)

**Fact:** He has worked as a researcher at various organizations.

**Ground Truth:** NS

**My Prediction:** S

**Category:** Context overgeneralization

**Why I got it wrong:**
The passage mentions Oliphant was an "Assistant Professor" doing research, and he "directed the BYU Biomedical Imaging Lab, and performed research on scanning impedance imaging." My model probably saw "performed research" and "various organizations" mentioned elsewhere and put them together. But here's the thing - being a professor who does research isn't the same as "working as a researcher." That's a different job title. The passage never actually says he was employed as a researcher at multiple places. It's a subtle distinction, and honestly, I can see why my model got confused - I had to read it twice myself to understand why this was marked NS.

## What I Learned

The main takeaway from this error analysis is that my model has opposite problems on each side:

**False positives happen** when my model is too generous with word meanings. It sees "engraver" and thinks "close enough to printmaker!" or sees someone doing research and concludes they're a "researcher." To fix this, I'd need stricter semantic matching - maybe penalize predictions when the terms aren't exact synonyms.

**False negatives happen** when my model is too literal. It needs the information stated in a straightforward way, and struggles when facts are embedded in phrases like "1995's Just Cause" or requires connecting two pieces of information. To fix this, I'd probably need better sentence parsing or maybe check multiple phrasings of the same fact.

The frustrating part is these two problems work against each other. When I lowered my threshold during tuning to catch more true positives, I got more false positives from the similar-terms issue. When I raised it to be more precise, I missed obvious cases. That's why I plateaued at 81.4% - it seems like the limit of what this approach can do without more fundamental changes to how the model processes the text.
