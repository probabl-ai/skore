<script lang="ts">
import EditableListItem from "@/components/EditableListItem.vue";

export interface EditableListItemModel {
  name: string;
  icon?: string;
  isUnnamed?: boolean;
  id: string;
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
      :key="item.name"
      v-model="items[index]"
      :actions="props.actions"
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
  padding: var(--spacing-padding-normal);
  gap: var(--spacing-padding-normal);
}
</style>
