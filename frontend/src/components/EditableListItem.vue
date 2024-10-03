<script setup lang="ts">
import { defineProps, nextTick, onMounted, ref, watch } from "vue";

import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import type { EditableListAction, EditableListItemModel } from "@/components/EditableList.vue";

const props = defineProps<{ actions?: EditableListAction[] }>();

const emit = defineEmits<{
  action: [payload: string];
  select: [];
  rename: [oldName: string, newName: string, item: EditableListItemModel];
}>();

const item = defineModel<EditableListItemModel>({ required: true });
const label = ref<HTMLSpanElement>();

function renameItem(newName: string) {
  const oldName = item.value.name;
  item.value.isNamed = true;
  item.value.name = newName;
  emit("rename", oldName, item.value.name, item.value);
}

function onEnter(e: Event) {
  (e.target as HTMLInputElement).blur();
}

function focusAndSelect() {
  if (label.value) {
    label.value.focus();

    const selection = window.getSelection();
    if (selection) {
      const range = document.createRange();
      range.selectNodeContents(label.value);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  }
}

function onBlur() {
  renameItem(label.value?.textContent ?? "unnamed");
}

watch(
  () => item.value.isNamed,
  async (isNamed) => {
    if (isNamed === false) {
      await nextTick();
      focusAndSelect();
    }
  }
);
onMounted(() => {
  if (!item.value.isNamed && label.value) {
    focusAndSelect();
  }
});
</script>

<template>
  <div class="editable-list-item">
    <div class="label-container" @click="emit('select')">
      <span class="icon" v-if="item.icon" :class="item.icon" />
      <span
        class="label"
        ref="label"
        :contenteditable="!item.isNamed"
        @keydown.enter="onEnter"
        @blur="onBlur"
      >
        {{ item.name }}
      </span>
    </div>
    <DropdownButton icon="icon-more" :is-inline="true" align="right">
      <DropdownButtonItem
        v-for="action in props.actions"
        :key="action.emitPayload"
        :label="action.label"
        :icon="action.icon"
        @click="emit('action', action.emitPayload)"
      />
    </DropdownButton>
  </div>
</template>

<style scoped>
.editable-list-item {
  display: flex;
  height: 25px;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-padding-small);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: var(--text-size-normal);
  font-weight: var(--text-weight-highlight);
  transition: background-color var(--transition-duration) var(--transition-easing);

  .icon {
    color: var(--color-secondary);
  }

  .label {
    min-width: 100px;
    color: var(--text-color-highlight);
    line-height: 2;
    outline: none;
    transition: font-weight var(--transition-duration) var(--transition-easing);

    &[contenteditable="true"] {
      background-color: var(--color-primary);
      caret-color: var(--color-secondary);
      color: var(--button-color);
      cursor: text;
    }
  }

  .icon:has(+ .label) {
    padding-right: var(--spacing-padding-small);
  }

  .dropdown {
    opacity: 0;
    transition: opacity var(--transition-duration) var(--transition-easing);
  }

  &:hover {
    background-color: var(--background-color-elevated);

    .label {
      font-weight: 500;
    }

    .dropdown {
      opacity: 1;
    }
  }
}
</style>
