<script setup lang="ts">
import { useTemplateRef } from "vue";

defineOptions({ inheritAttrs: false });
const model = defineModel<string>("value");

const textarea = useTemplateRef<HTMLTextAreaElement>("textarea");
let selectionStart = -1;
let selectionEnd = -1;
let selectedText = "";

function updateSelections() {
  if (textarea.value && model.value) {
    const start = textarea.value.selectionStart;
    const end = textarea.value.selectionEnd;

    selectionStart = start;
    selectionEnd = end;
    selectedText = model.value.substring(start, end);
  }
}

function replaceSelectedTextWith(value: string) {
  if (textarea.value && model.value && selectionStart !== -1 && selectionEnd !== -1) {
    const text = model.value;
    const before = text.substring(0, selectionStart);
    const after = text.substring(selectionEnd);

    model.value = before + value + after;
  }
}

function markBold() {
  replaceSelectedTextWith(`**${selectedText}**`);
}

function markItalic() {
  replaceSelectedTextWith(`*${selectedText}*`);
}

function getLineIndicesFromBounds(text: string, start: number, end: number): [number, number] {
  // Get text before selection start
  const textBeforeStart = text.substring(0, start);
  // Start line is number of newlines before selection
  const startLine = textBeforeStart.split("\n").length - 1;

  // Get text from start to end of selection
  const selectionText = text.substring(start, end);
  // End line is start line plus newlines in selection
  const endLine = startLine + selectionText.split("\n").length - 1;

  return [startLine, endLine];
}

function markList() {
  if (model.value) {
    const lines = model.value.split("\n");
    const [startLine, endLine] = getLineIndicesFromBounds(
      model.value,
      selectionStart,
      selectionEnd
    );

    const list = lines.map((l, i) => (i >= startLine && i <= endLine ? `- ${l}` : l));
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
    <!-- bind $attrs to
    pass attribute like col, rows, ...
    and make all native event bubble -->
    <textarea
      ref="textarea"
      v-model="model"
      v-bind="$attrs"
      @mouseup="updateSelections"
      @keyup="updateSelections"
    ></textarea>
  </div>
</template>

<style lang="css" scoped>
.rich-text-editor {
  width: 100%;
  height: 100%;

  & textarea {
    width: 100%;
    height: 100%;
    border: none;
    background-color: var(--color-background-primary);
    color: var(--color-text-secondary);
    font-family: GeistMono, monospace;
    resize: none;

    &:focus {
      outline: none;
    }
  }
}
</style>
