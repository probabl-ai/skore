<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, useTemplateRef, watch } from "vue";

import RichTextEditor from "@/components/RichTextEditor.vue";
import SimpleButton from "@/components/SimpleButton.vue";
import { useActivityStore } from "@/views/activity/activity";

const props = defineProps<{ name: string; note: string | null; version: number }>();
const emit = defineEmits(["editionEnd"]);
defineExpose({ focus });

const activityStore = useActivityStore();
const el = useTemplateRef<HTMLDivElement>("el");
const editor = useTemplateRef<InstanceType<typeof RichTextEditor>>("editor");
const isEditing = ref(false);
const innerNote = ref(`${props.note !== null ? props.note : ""}`);

function focus() {
  isEditing.value = true;
  editor.value?.focus();
  // start to listen for click outside of this component
  document.addEventListener("click", onClickOutside);
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
    // actually wait for the editor to be closed to restart backend polling
    nextTick(() => {
      activityStore.startBackendPolling();
      emit("editionEnd");
    });
  }
}

async function onClickOutside(e: Event) {
  if (isEditing.value) {
    const clicked = e.target as Node;
    if (el.value && document.body.contains(clicked)) {
      // is it a click outside ?
      const isOutside = !el.value.contains(clicked);
      console.log("clicked outside", isOutside);
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
  <div class="item-note-editor" ref="el" :class="{ focus: isEditing }">
    <!-- <Transition name="fade"> -->
    <div class="edit-actions">
      <SimpleButton :is-inline="true" icon="icon-bold" @click="editor?.markBold()" />
      <SimpleButton :is-inline="true" icon="icon-italic" @click="editor?.markItalic()" />
      <SimpleButton :is-inline="true" icon="icon-bullets" @click="editor?.markList()" />
    </div>
    <!-- </Transition> -->
    <RichTextEditor
      class="inline-editor"
      ref="editor"
      v-model:value="innerNote"
      :rows="4"
      @keyup.esc="onEditionEnd"
      @keydown.shift.enter.prevent="onEditionEnd"
      @keydown.meta.enter.prevent="onEditionEnd"
    />
  </div>
</template>

<style scoped>
.item-note-editor {
  position: relative;
  padding-top: var(--spacing-8);
  border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  border-radius: var(--radius-xs);
  color: var(--color-text-secondary);
  transition: border-color var(--animation-duration) var(--animation-easing);

  .edit-actions {
    position: absolute;
    top: 0;
    right: 0;
    display: flex;
    flex-direction: row;
    justify-content: flex-end;
  }

  .inline-editor {
    padding-top: var(--spacing-12);
  }

  &.focus {
    border-color: var(--color-stroke-background-branding);
  }
}
</style>
