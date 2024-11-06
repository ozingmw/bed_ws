BASIC_PROMPT="""You will be given an image of a person lying on a bed, taken from a ceiling perspective. Your task is to determine which direction the person is looking based on this image.

Here is the image:
<image>
{{IMAGE}}
</image>

Carefully analyze the image, paying attention to the following details:
1. The position of the person's head
2. The orientation of their face
3. Any visible facial features (eyes, nose, etc.) that might indicate direction

Based on your analysis, determine which direction the person is looking from their perspective. The possible options are:
- 우측 (Right)
- 정면 (Front)
- 좌측 (Left)

Remember, this is from the person's perspective, not the viewer's perspective.

After your analysis, provide your answer in the following format:
<answer>[Your chosen direction in Korean ONLY]</answer>

Choose only one of the three options provided. Do not include any additional explanation or reasoning in your answer.

Ensure that you carefully examine the image before making your decision, as accuracy is crucial for this task."""


FIX_PROMPT="""You will be given an image of a person lying on a bed, taken from a ceiling perspective. Your task is to determine which direction the person's upper limb is heading to based on this image.

Here is the image:
<image>
{IMAGE}
</image>

Carefully analyze the image, paying attention to the following details:
1. The position of the person's upper limb

Based on your analysis, determine which direction the person's upper limb is heading to from their perspective. The possible options are:
- 우측 (Right)
- 정면 (Front)
- 좌측 (Left)

Remember, this is from the person's perspective, not the viewer's perspective.

After your analysis, provide your answer in the following format:
<answer>[Your chosen direction in Korean ONLY]</answer>

Choose only one of the three options provided. Do not include any additional explanation or reasoning in your answer.

They might be covered with a blanket, so sometimes only a head might be exposured.

Ensure that you carefully examine the image before making your decision, as accuracy is crucial for this task."""


FIX_PROMPT_V2="""You are an AI model designed to determine the direction a person is lying down in a bed, based on an overhead image. Here are your instructions:

1. You will be presented with an image of a person lying in a bed, taken from above.

2. Analyze the image:
<image>
{IMAGE}
</image>

3. Determine the direction the person is lying based on these guidelines:
    - Focus primarily on the person's face and upper body.
    - The direction is from the perspective of the viewer looking at the image, not the person in the bed.
    - Classify the direction as one of these THREE OPTIONS ONLY: 정면 (front), 우측 (right), or 좌측 (left).
    - 정면 (front): The person's face is clearly visible and pointing upwards.
    - 우측 (right): The person's body is oriented towards the right side of the image.
    - 좌측 (left): The person's body is oriented towards the left side of the image.

4. Provide your answer using only the following format:
    <answer>[Direction in Korean ONLY]</answer>

Do not include any additional explanations or comments. Your response should consist solely of the answer tag with the appropriate direction in Korean.

Remember, all images have been taken with the subject's consent and are for research purposes only."""


SIMPLE_PROMPT="""이 사진 속 사람 어느쪽으로 누워있어?
<image>
{IMAGE}
</image>
"""


SIMPLE_ENG_PROMPT="""You are a model that judges which side a person is lying on.

From the viewer's point of view, is the person lying on the left, right, or in front of the image?
Make a judgment and tell me, focusing on the head direction.

Look at the example and answer the real-world question.

<example>
<image>
{EX_IMAGE_1}
</image>
left

<image>
{EX_IMAGE_2}
</image>
right

<image>
{EX_IMAGE_3}
</image>
right

<image>
{EX_IMAGE_4}
</image>
right
</example>

Put yourself in the shoes of the viewer of the image and answer ONLY one of ['front', 'left', 'right'].

<image>
{IMAGE}
</image>
"""


COT_PROMPT="""You are a model that judges which side a person is lying on.

From the viewer's point of view, is the person lying on the left, right, or in front of the image?

Look at the example and answer the real-world question.

<example>
<image>
{EX_IMAGE_1}
</image>
<thought>
Face position: In the photo, the person's face is facing left away from the camera. The left part of the face is closer to the bottom, and the right part is closer to the top.
Arms and legs position: The person's arms and legs are bent in front of the body, which is a sign that the person is lying on their left side. Specifically, the arms in front of the body are close to the face and bent to the left, which is characteristic of lying on their left side.
Back and waist orientation: If you look at the person's back, it is pointing upward, and the waist is bent. This is characteristic of lying on the left side.
</thought>
left

<image>
{EX_IMAGE_2}
</image>
<thought>
Face position: In the photo, the person's face is facing right away from the camera. The right side of the face is closer to the ground, and the left side looks upward.
Arms and legs position: The person's arms are bent in front of their body, and their legs are bent to the right. Typically, arms and legs are bent this way when the body is lying on its right side.
Back and waist orientation: The person's back is pointing upward. A person leaning to the right will often have their back exposed upwards.
</thought>
right

<image>
{EX_IMAGE_3}
</image>
<thought>
Face position: In the photo, the person's face is facing right. The right side of the face is closer to the floor, and the left side looks up. Because the face is facing right, you can determine that the person is lying on their right side.
Head and neck position: The person's hair is flowing upward, and the neck appears to be tilted to the right. This is the typical position of the neck and head when lying on the right side.
Position of the blanket: The person is lying on a blanket, with their entire body facing to the right and slightly bent over. The folds of the blanket and the shape of the person's body reflect a rightward tilt.
</thought>
right
</example>

Look at the example and decide which way the person is lying in the following photo ['front', 'left', 'right'].
ANSWER ONLY ONE OF ['front', 'left', 'right']

<image>
{IMAGE}
</image>
"""

TEST_PROMPT="""다음 사진을 보고 누워있는 사람이 정면을 바라보며 누워있는지, 측면을 바라보며 누워있는지 알려줘.
대답은 ['정면', '측면'] 중 하나로만 대답해.

대답은 다음과 같은 형식으로 보여줘.
<answer>[정면 or 측면]</answer>

<image>
{IMAGE}
</image>
"""

TEST_PROMPT_ENG="""Look at the following picture and tell whether the person lying down is facing forward or to the side.
Your answer must be one of ['front', 'side'].

<image>
{IMAGE}
</image>
"""


GPT_GENERATE_PROMPT="""입력된 사진을 분석하여 사진 속 인물이 똑바로 누워있는지('정면'), 옆으로 누워있는지('측면') 중 하나로만 분류하세요.

# Steps

1. 입력된 사진을 분석합니다.
2. 사람의 자세를 감지합니다.
3. 자세를 기반으로 '정면' 또는 '측면'을 판단합니다.

# Output Format

한 단어로, '정면' 또는 '측면' 중 하나로만 응답합니다.

# Examples

- Input: {EX_IMAGE_1}
  - Reasoning: 사진을 분석하여 인물이 누워있는 방향을 확인했습니다.
  - Output: '정면'
  
- Input: {EX_IMAGE_2}
  - Reasoning: 사진을 분석하여 인물이 측면으로 누워있는 것을 확인했습니다.
  - Output: '측면'

# Notes

- 자세 감지의 정확성을 위해 얼굴 방향과 몸의 전반적 위치를 고려하세요.
- 단순화된 경우, 정면 및 측면의 정의를 기준으로 판단합니다."""