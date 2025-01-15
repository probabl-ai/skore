<script setup lang="ts">
import { ref, useTemplateRef } from "vue";

const model = defineModel<string>("value");

const textarea = useTemplateRef<HTMLTextAreaElement>("textarea");
const selectionStart = ref(-1);
const selectionEnd = ref(-1);
const selectedText = ref("");

function updateSelections() {
  if (textarea.value && model.value) {
    const start = textarea.value.selectionStart;
    const end = textarea.value.selectionEnd;

    selectionStart.value = start;
    selectionEnd.value = end;
    selectedText.value = model.value.substring(start, end);
  }
}

function replaceSelectedTextWith(value: string) {
  if (textarea.value && model.value && selectionStart.value !== -1 && selectionEnd.value !== -1) {
    const text = model.value;
    const before = text.substring(0, selectionStart.value);
    const after = text.substring(selectionEnd.value);

    model.value = before + value + after;
  }
}

function markBold() {
  const selection = selectedText.value;
  replaceSelectedTextWith(`**${selection}**`);
}

function markItalic() {
  const selection = selectedText.value;
  replaceSelectedTextWith(`*${selection}*`);
}

function markList() {
  if (model.value) {
    const lines = model.value.split("\n");
    const textBeforeSelection = model.value.substring(0, selectionStart.value);
    const selectedLineIndex = textBeforeSelection.split("\n").length - 1;
    const list = lines.map((l, i) => (i === selectedLineIndex ? `- ${l}` : l));
    model.value = list.join("\n");
  }
}

function focus() {
  textarea.value?.focus();
}

defineExpose({ markBold, markItalic, markList, focus });
</script>

<template>
  <div class="rich-text-editor">
    <textarea ref="textarea" v-model="model" v-bind="$attrs" @mouseup="updateSelections"></textarea>
  </div>
</template>

<style lang="css" scoped>
.rich-text-editor {
  width: 100%;
  height: 100%;

  & textarea {
    width: 100%;
    height: 100%;
    border: var(--stroke-width-md) solid var(--color-stroke-background-primary);
    border-radius: var(--radius-xs);
    color: var(--color-text-secondary);
    font-family: GeistMono, monospace;
    resize: none;
    transition: border-color var(--animation-duration) var(--animation-easing);

    &:focus {
      border-color: var(--color-stroke-background-branding);
      outline: none;
    }
  }
}
</style>
