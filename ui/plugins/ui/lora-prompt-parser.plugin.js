/*
    LoRA Prompt Parser 1.0
    by Patrice

    Copying and pasting a prompt with a LoRA tag will automatically select the corresponding option in the Easy Diffusion dropdown and remove the LoRA tag from the prompt. The LoRA must be already available in the corresponding Easy Diffusion dropdown (this is not a LoRA downloader).
*/
(function() {
    "use strict"
    
    promptField.addEventListener('input', function(e) {
        let loraExtractSetting = document.getElementById("extract_lora_from_prompt")
        if (!loraExtractSetting.checked) {
            return
        }

        const { LoRA, prompt } = extractLoraTags(e.target.value);
		//console.log('e.target: ' + JSON.stringify(LoRA));
    
        if (LoRA !== null && LoRA.length > 0) {
            promptField.value = prompt.replace(/,+$/, ''); // remove any trailing ,
    
            if (testDiffusers?.checked === false) {
                showToast("LoRA's are only supported with diffusers. Just stripping the LoRA tag from the prompt.")
            }
        }
                                 
        if (LoRA !== null && LoRA.length > 0 && testDiffusers?.checked) {
            let modelNames = LoRA.map(e => e.lora_model_0)
            let modelWeights = LoRA.map(e => e.lora_alpha_0)
            loraModelField.value = {modelNames: modelNames, modelWeights: modelWeights}

            showToast("Prompt successfully processed")
			
        }
            
        //promptField.dispatchEvent(new Event('change'));
    });
    
    // extract LoRA tags from strings
    function extractLoraTags(prompt) {
        // Define the regular expression for the tags
        const regex = /<(?:lora|lyco):([^:>]+)(?::([^:>]*))?(?::([^:>]*))?>/gi

        // Initialize an array to hold the matches
        let matches = []

        // Iterate over the string, finding matches
        for (const match of prompt.matchAll(regex)) {
            const modelFileName = match[1].trim()
            const loraPathes = getAllModelPathes("lora", modelFileName)
            if (loraPathes.length > 0) {
                const loraPath = loraPathes[0]
                // Initialize an object to hold a match
                let loraTag = {
                    lora_model_0: loraPath,
                }
				//console.log("Model:" +  modelFileName);
        
                // If weight is provided, add it to the loraTag object
                if (match[2] !== undefined && match[2] !== '') {
                    loraTag.lora_alpha_0 = parseFloat(match[2].trim())
                }
                else
                {
                    loraTag.lora_alpha_0 = 0.5
                }
				
        
                // If blockweights are provided, add them to the loraTag object
                if (match[3] !== undefined && match[3] !== '') {
                    loraTag.blockweights = match[3].trim()
                }
        
                // Add the loraTag object to the array of matches
                matches.push(loraTag);
				//console.log(JSON.stringify(matches));
            }
            else
            {
                showToast("LoRA not found: " + match[1].trim(), 5000, true)            
            }
        }

        // Clean up the prompt string, e.g. from "apple, banana, <lora:...>, orange, <lora:...>  , pear <lora:...>, <lora:...>" to "apple, banana, orange, pear"
        let cleanedPrompt = prompt.replace(regex, '').replace(/(\s*,\s*(?=\s*,|$))|(^\s*,\s*)|\s+/g, ' ').trim();
		//console.log('Matches: ' + JSON.stringify(matches));

        // Return the array of matches and cleaned prompt string
        return {
            LoRA: matches,
            prompt: cleanedPrompt
        }
    }
})()
