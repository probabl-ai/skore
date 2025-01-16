<script setup lang="ts">
import { ref, useTemplateRef, watch } from "vue";

import MarkdownWidget from "@/components/MarkdownWidget.vue";
import RichTextEditor from "@/components/RichTextEditor.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import type { PresentableItem } from "@/models";
import { debounce } from "@/services/utils";
import { useProjectStore } from "@/stores/project";

const props = defineProps<{ item: PresentableItem }>();

const projectStore = useProjectStore();
const isEditing = ref(false);
const editor = useTemplateRef<InstanceType<typeof RichTextEditor>>("editor");

const richText = ref(`${props.item.note ?? ""}`);
const debouncedSetNote = debounce(
  () => {
    projectStore.setNoteOnItem(props.item.name, richText.value);
  },
  500,
  true
);

watch(richText, () => {
  debouncedSetNote();
});
</script>

<template>
  <div class="item-note">
    <div class="header" @click="isEditing = false">
      <div>Note</div>
      <Transition name="fade">
        <div class="edit-actions" v-if="isEditing">
          <SimpleButton :is-inline="true" icon="icon-bold" @click="editor?.markBold()" />
          <SimpleButton :is-inline="true" icon="icon-italic" @click="editor?.markItalic()" />
          <SimpleButton :is-inline="true" icon="icon-bullets" @click="editor?.markList()" />
        </div>
      </Transition>
    </div>
    <div class="editor" v-if="isEditing">
      <RichTextEditor ref="editor" v-model:value="richText" :rows="1" />
    </div>
    <div
      class="preview"
      v-if="!isEditing"
      @click="
        isEditing = true;
        editor?.focus();
      "
    >
      <MarkdownWidget :source="richText" />
    </div>
  </div>
</template>

<style lang="css" scoped>
.item-note {
  border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  border-radius: var(--radius-xs);
  color: var(--color-text-secondary);

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--spacing-8);
    border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    background-color: var(--color-background-secondary);

    .edit-actions {
      display: flex;
      flex-direction: row;
      justify-content: flex-end;
    }
  }

  .preview {
    padding: var(--spacing-8);
  }
}
</style>
