<script lang="ts">
import EditableListItem from "@/components/EditableListItem.vue";

export interface EditableListItemProps {
  name: string;
  icon?: string;
}

export interface EditableListAction {
  label: string;
  emitPayload: string;
  icon?: string;
}
</script>

<script setup lang="ts">
const props = defineProps<{
  items: EditableListItemProps[];
  actions?: EditableListAction[];
}>();

const emit = defineEmits<{ action: [payload: string, item: EditableListItemProps] }>();

const onAction = (payload: string, item: EditableListItemProps) => {
  emit("action", payload, item);
};
</script>

<template>
  <div class="editable-list">
    <div class="editable-list-item" v-for="item in props.items" :key="item.name">
      <EditableListItem :item="item" :actions="props.actions" @action="onAction($event, item)" />
    </div>
  </div>
</template>

<style scoped>
.editable-list {
  display: flex;
  flex-direction: column;
  padding: var(--spacing-padding-normal);
  gap: var(--spacing-padding-normal);
}
</style>
