<script lang="ts">
import EditableListItem from "@/components/EditableListItem.vue";

export interface EditableListItemModel {
  id: string;
  name: string;
  icon?: string;
  isNamed?: boolean;
  isSelected?: boolean;
}

export interface EditableListAction {
  label: string;
  emitPayload: string;
  icon?: string;
}
</script>

<script setup lang="ts">
const props = defineProps<{
  actions?: EditableListAction[];
}>();

const emit = defineEmits<{
  action: [payload: string, item: EditableListItemModel];
  select: [name: string];
  rename: [oldName: string, newName: string, item: EditableListItemModel];
}>();

const items = defineModel<EditableListItemModel[]>("items", { required: true });

function onAction(payload: string, item: EditableListItemModel) {
  emit("action", payload, item);
}

function onRename(oldName: string, newName: string, item: EditableListItemModel) {
  emit("rename", oldName, newName, item);
}
</script>

<template>
  <div class="editable-list">
    <EditableListItem
      v-for="(item, index) in items"
      :key="item.id"
      v-model="items[index]"
      :actions="props.actions"
      :is-selected="item.isSelected"
      @action="onAction($event, item)"
      @select="emit('select', item.name)"
      @rename="onRename"
    />
  </div>
</template>

<style scoped>
.editable-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-4);
}
</style>
