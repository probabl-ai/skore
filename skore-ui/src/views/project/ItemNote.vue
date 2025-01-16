<script setup lang="ts">
import { nextTick, ref, useTemplateRef, watch } from "vue";

import MarkdownWidget from "@/components/MarkdownWidget.vue";
import RichTextEditor from "@/components/RichTextEditor.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import { debounce } from "@/services/utils";
import { useProjectStore } from "@/stores/project";

const props = defineProps<{ name: string; note: string | null }>();

const projectStore = useProjectStore();
const editor = useTemplateRef<InstanceType<typeof RichTextEditor>>("editor");
const isEditing = ref(false);
const innerNote = ref(`${props.note !== null ? props.note : ""}`);

const debouncedSetNote = debounce(
  () => {
    projectStore.setNoteOnItem(props.name, innerNote.value);
  },
  500,
  true
);

function onEdit() {
  isEditing.value = true;
  nextTick(() => {
    editor.value?.focus();
  });
}

function onEditionEnd() {
  debouncedSetNote();
  isEditing.value = false;
}

function onClear() {
  innerNote.value = "";
  onEditionEnd();
}

watch(innerNote, () => {
  debouncedSetNote();
});

watch(
  () => props.note,
  (newNote) => {
    innerNote.value = `${newNote !== null ? newNote : ""}`;
  }
);
</script>

<template>
  <div class="item-note">
    <div class="header">
      <div class="info">
        <div>Note</div>
        <Transition name="fade">
          <SimpleButton
            v-if="isEditing && innerNote && innerNote.length > 0"
            label="clear"
            class="clear"
            :is-inline="true"
            @click="onClear"
          />
        </Transition>
      </div>
      <Transition name="fade">
        <div class="edit-actions" v-if="isEditing">
          <SimpleButton :is-inline="true" icon="icon-bold" @click="editor?.markBold()" />
          <SimpleButton :is-inline="true" icon="icon-italic" @click="editor?.markItalic()" />
          <SimpleButton :is-inline="true" icon="icon-bullets" @click="editor?.markList()" />
        </div>
      </Transition>
    </div>
    <div class="editor" v-if="isEditing">
      <RichTextEditor
        ref="editor"
        v-model:value="innerNote"
        :rows="4"
        @keyup.esc="onEditionEnd"
        @keydown.shift.enter="onEditionEnd"
        @keydown.meta.enter="onEditionEnd"
      />
    </div>
    <div class="preview" v-if="!isEditing" @click="onEdit">
      <MarkdownWidget :source="innerNote" v-if="innerNote && innerNote.length > 0" />
      <div class="placeholder" v-else>Click to annotate {{ props.name }}.</div>
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

    .info {
      display: flex;
      gap: var(--spacing-8);

      .clear {
        padding: 0;
        color: var(--color-text-danger);
        font-size: var(--font-size-xs);
      }
    }

    .edit-actions {
      display: flex;
      flex-direction: row;
      justify-content: flex-end;
    }
  }

  .preview {
    padding: var(--spacing-8);

    .placeholder {
      color: var(--color-text-secondary);
      font-size: var(--font-size-xs);
      font-style: italic;
    }
  }
}
</style>
