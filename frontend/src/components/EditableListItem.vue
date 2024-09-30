<script setup lang="ts">
import { defineProps } from "vue";

import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import type { EditableListAction, EditableListItemProps } from "@/components/EditableList.vue";

const props = defineProps<{ item: EditableListItemProps; actions?: EditableListAction[] }>();

const emit = defineEmits<{ action: [payload: string] }>();
</script>

<template>
  <div class="editable-list-item">
    <div class="label-container">
      <span class="icon" v-if="props.item.icon" :class="props.item.icon" />
      <span class="label">
        {{ props.item.name }}
      </span>
    </div>
    <DropdownButton icon="icon-equal" :is-inline="true" align="right">
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
  font-size: var(--text-size-normal);
  font-weight: var(--text-weight-normal);
  transition: background-color var(--transition-duration) var(--transition-easing);

  .icon {
    color: var(--color-secondary);
  }

  .label {
    color: var(--color-text-highlight);
    transition: font-weight var(--transition-duration) var(--transition-easing);
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
