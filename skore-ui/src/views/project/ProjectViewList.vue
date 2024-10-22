<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { onMounted, ref } from "vue";

import type { EditableListItemModel } from "@/components/EditableList.vue";
import EditableList from "@/components/EditableList.vue";
import SectionHeader from "@/components/SectionHeader.vue";
import { generateRandomId } from "@/services/utils";
import { useProjectStore } from "@/stores/project";
import { useToastsStore } from "@/stores/toasts";

const projectStore = useProjectStore();
const toastsStore = useToastsStore();
const views = ref<EditableListItemModel[]>([]);
let unsavedViewsId: string = "";

function onViewSelected(view: string) {
  projectStore.setCurrentView(view);
}

async function onViewRenamed(oldName: string, newName: string, item: EditableListItemModel) {
  const viewAlreadyExists = views.value.filter((view) => view.name === newName).length > 1;
  if (viewAlreadyExists) {
    toastsStore.addToast(`A view named "${newName}" already exists`, "error");
    item.isNamed = false;
    return;
  }
  // user can rename an existing view
  // and rename a new view that is not known by the backend
  if (unsavedViewsId == item.id) {
    await projectStore.createView(newName);
    unsavedViewsId = "";
    toastsStore.addToast("View added successfully", "success", { duration: 20 });
  } else {
    await projectStore.renameView(oldName, newName);
  }
  // auto switch the view to the renamed one
  onViewSelected(newName);
}

async function onViewsListAction(action: string, item: EditableListItemModel) {
  switch (action) {
    case "rename": {
      item.isNamed = false;
      break;
    }
    case "duplicate": {
      const index = views.value.indexOf(item) ?? 0;
      const newName = `${item.name} - copy`;
      views.value.splice(index + 1, 0, {
        name: newName,
        icon: "icon-new-document",
        isNamed: false,
        id: generateRandomId(),
      });
      unsavedViewsId = newName;
      await projectStore.duplicateView(item.name, newName);
      break;
    }
    case "delete": {
      views.value.splice(views.value.indexOf(item), 1);
      await projectStore.deleteView(item.name);
      toastsStore.addToast("View deleted successfully", "success", { duration: 20 });
      break;
    }
  }
}

function onAddView() {
  const hasUnsavedViews = views.value.some((view) => !view.isNamed);
  if (!hasUnsavedViews) {
    const id = generateRandomId();
    views.value.push({ name: "New view", icon: "icon-new-document", isNamed: false, id });
    unsavedViewsId = id;
  }
}

onMounted(async () => {
  views.value = Object.keys(projectStore.views).map((key) => ({
    name: key,
    icon: "icon-new-document",
    isNamed: true,
    id: generateRandomId(),
  }));
});
</script>

<template>
  <div class="views-list">
    <SectionHeader
      title="Views"
      icon="icon-recent-document"
      action-icon="icon-plus"
      action-tooltip="Add a new view"
      @action="onAddView"
    />
    <Simplebar class="list">
      <EditableList
        v-model:items="views"
        :actions="[
          { label: 'rename', emitPayload: 'rename', icon: 'icon-edit' },
          { label: 'duplicate', emitPayload: 'duplicate', icon: 'icon-copy' },
          { label: 'delete', emitPayload: 'delete', icon: 'icon-trash' },
        ]"
        @action="onViewsListAction"
        @select="onViewSelected"
        @rename="onViewRenamed"
      />
    </Simplebar>
  </div>
</template>

<style scoped>
.views-list {
  background-color: var(--background-color-elevated-high);

  & .list {
    max-height: 20vh;
  }
}
</style>
