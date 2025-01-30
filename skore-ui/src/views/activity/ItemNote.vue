<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, useTemplateRef, watch } from "vue";

import MarkdownWidget from "@/components/MarkdownWidget.vue";
import RichTextEditor from "@/components/RichTextEditor.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import { useActivityStore } from "@/views/activity/activity";

const props = defineProps<{ name: string; note: string | null; version: number }>();

const activityStore = useActivityStore();
const el = useTemplateRef<HTMLDivElement>("el");
const editor = useTemplateRef<InstanceType<typeof RichTextEditor>>("editor");
const isEditing = ref(false);
const innerNote = ref(`${props.note !== null ? props.note : ""}`);

function onEdit() {
  activityStore.stopBackendPolling();
  isEditing.value = true;
  // start to listen for click outside of this component
  document.addEventListener("click", onClickOutside);
  // actually wait for the editor to be open to focus it
  nextTick(() => {
    editor.value?.focus();
  });
}

async function onEditionEnd() {
  // if there is multiple instances of the component in the page
  // this function may be called multiple times
  // so guard this call
  if (isEditing.value) {
    // stop listening to outside click
    document.removeEventListener("click", onClickOutside);
    innerNote.value = innerNote.value.replace(/\n+$/g, "");
    await activityStore.setNoteOnItem(props.name, props.version, innerNote.value);
    isEditing.value = false;
    // actually wait for the editor to be closed to resatrt backend polling
    nextTick(() => {
      activityStore.startBackendPolling();
    });
  }
}

async function onClear() {
  innerNote.value = "";
  await onEditionEnd();
}

async function onClickOutside(e: Event) {
  if (isEditing.value) {
    const clicked = e.target as Node;
    if (el.value && document.body.contains(clicked)) {
      // is it a click outside ?
      const isOutside = !el.value.contains(clicked);
      if (isOutside) {
        await onEditionEnd();
      }
    }
  }
}

watch(innerNote, async () => {
  await activityStore.setNoteOnItem(props.name, props.version, innerNote.value);
});

watch(
  () => props.note,
  (newNote) => {
    innerNote.value = `${newNote !== null ? newNote : ""}`;
  }
);

onBeforeUnmount(() => {
  // avoid event listener leak in case the component is unmounted in edit mode
  document.removeEventListener("click", onClickOutside);
});
</script>

<template>
  <div class="item-note" ref="el">
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
        @keydown.shift.enter.prevent="onEditionEnd"
        @keydown.meta.enter.prevent="onEditionEnd"
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
