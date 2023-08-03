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
            for (let i = 0; i < LoRA.length; i++) {
            //if (loraModelField.value !== LoRA[0].lora_model) {
                // Set the new LoRA value
				//console.log("Loading info");
				//console.log(LoRA[0].lora_model_0);
				//console.log(JSON.stringify(LoRa));
				
                let lora = `lora_model_${i}`;
                let alpha = `lora_alpha_${i}`;
                let loramodel = document.getElementById(lora);
                let alphavalue = document.getElementById(alpha);
				loramodel.setAttribute("data-path", LoRA[i].lora_model_0);
                loramodel.value = LoRA[i].lora_model_0;
                alphavalue.value = LoRA[i].lora_alpha_0;
                if (i != LoRA.length - 1)
                    createLoraEntry();
            }
                //loraAlphaSlider.value = loraAlphaField.value * 100;
                //TBD.value = LoRA[0].blockweights; // block weights not supported by ED at this time
            //}
            showToast("Prompt successfully processed", LoRA[0].lora_model_0);
			//console.log('LoRa: ' + LoRA[0].lora_model_0);
			//showToast("Prompt successfully processed", lora_model_0.value);
			
        }
            
        //promptField.dispatchEvent(new Event('change'));
    });
    
    function isModelAvailable(array, searchString) {
        const foundItem = array.find(function(item) {
            item = item.toString().toLowerCase();
            return item === searchString.toLowerCase()
        });

        return foundItem || "";
    }

    // extract LoRA tags from strings
    function extractLoraTags(prompt) {
        // Define the regular expression for the tags
        const regex = /<(?:lora|lyco):([^:>]+)(?::([^:>]*))?(?::([^:>]*))?>/gi

        // Initialize an array to hold the matches
        let matches = []

        // Iterate over the string, finding matches
        for (const match of prompt.matchAll(regex)) {
            const modelFileName = isModelAvailable(modelsCache.options.lora, match[1].trim())
            if (modelFileName !== "") {
                // Initialize an object to hold a match
                let loraTag = {
                    lora_model_0: modelFileName,
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
