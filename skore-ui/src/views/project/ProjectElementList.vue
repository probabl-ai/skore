<script setup lang="ts">
import Simplebar from "simplebar-vue";

import SectionHeader from "@/components/SectionHeader.vue";
import TreeAccordion from "@/components/TreeAccordion.vue";
import { useProjectStore } from "@/stores/project";
import { useToastsStore } from "@/stores/toasts";

const projectStore = useProjectStore();
const toastsStore = useToastsStore();

function onItemSelected(key: string) {
  if (projectStore.currentView) {
    projectStore.displayKey(projectStore.currentView, key);
  } else {
    toastsStore.addToast("No view selected", "error");
  }
}
</script>

<template>
  <div class="keys-list">
    <SectionHeader title="Items" icon="icon-pie-chart" />
    <Simplebar class="scrollable">
      <TreeAccordion :nodes="projectStore.keysAsTree()" @item-selected="onItemSelected" />
    </Simplebar>
  </div>
</template>

<style scoped>
.keys-list {
  display: flex;
  max-height: 100%;
  flex-direction: column;

  & .scrollable {
    height: 0;
    flex: 1;
    padding: var(--spacing-padding-large);
    background-color: var(--background-color-elevated-high);
  }
}
</style>
