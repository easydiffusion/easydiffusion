import { API_GENERATE_HEALTH, API_GENERATE_IMAGE_URL, GeneratedImageData } from "./constants"

export type GeneratedImageDataPayload = {
  input: {
    prompt: string;
    num_outputs: number,
    num_inference_steps: number,
    width: number,
    height: number,
    seed: number,
    guidance_scale: number,
  }
}

export type GenerateImageResponseType = {
  status: string;
  output: string[];
  filename: string;
}

// turn GeneratedImageData into GeneratedImageDataPayload
export const buildGenerateImagePayload = (data: GeneratedImageData) => {
  return {
    guidance_scale: parseFloat(data.guidance),
    height: data.height,
    num_inference_steps: data.steps,
    num_outputs: 1,
    prompt: data.prompt.finalPrompt,
    prompt_text: data.prompt.text,
    prompt_modifiers: data.prompt.modifiers,
    seed: parseInt(data.seed) || '-1',
    width: data.width,
  }
}

export const generateImage = async (data: GeneratedImageData) => {
  try {
    // @ts-expect-error This is to allow the user to override the API endpoint
    const response = await fetch(window.SD_URL || API_GENERATE_IMAGE_URL, {
      method: 'POST',
      mode: "cors",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(buildGenerateImagePayload(data)),
    })

    console.log('generateImage response', response);
    const responseData: GenerateImageResponseType = await response.json();
    // console.log('responseData', responseData);
    console.log('responseData.output', responseData.output[0].data);

    const imageUrl = responseData.filename;
    return imageUrl;
  } catch (error) {
    console.log(error)
  }
}

export const checkServer = async () => {
  try {
    // @ts-expect-error This is to allow the user to override the API endpoint
    const response = await fetch(window.SD_URL || API_GENERATE_HEALTH);
    const responseData = await response.json();
    return responseData[0] === 'OK';
  } catch (error) {
    console.log(error)
    return false;
  }
}