<template>
  <n-tag class="pl-0 pr-2" :color="{ color: '', textColor: '', borderColor: '#0043ae'}" v-if="!editing">
    <div class="flex items-center">
      <div class="drag-handle pl-1 pr-2 hover:(text-blue-200)" style="cursor: grab;">
        <Draggable class="w-5 h-5" />
      </div>
      <div class="modifier-tag text-[13px]" :title="element" @click="editing = !editing">
        {{ element }}
      </div>
      <div @click="removeModifier(element)" class="ml-2 cursor-pointer hover:text-cyan-500 transition-colors">
        <Close class="w-5 h-5" />
      </div>
    </div>
  </n-tag>
  <div v-else>
    <n-input
      v-model:value="newElement"
      type="text"
      size="small"
      @blur="sendUpdate"
      @keydown.enter="sendUpdate"
    />
  </div>
</template>
<script setup lang="ts">
  import {
  Close,
  Draggable,
} from "@vicons/carbon";
import {
  NTag,
  NInput,
} from "naive-ui";
import { ref } from "vue";

const props = defineProps<{
  element: string;
  removeModifier: (element: string) => void;
}>();
const emit = defineEmits(["update:element", "remove:element"]);

const editing = ref(false);
const newElement = ref(props.element);

const sendUpdate = () => {
  editing.value = false;
  emit("update:element", [newElement.value, props.element]);
};


</script>