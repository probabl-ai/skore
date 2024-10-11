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
  <SectionHeader title="Elements" icon="icon-pie-chart" />
  <Simplebar class="key-list">
    <TreeAccordion :nodes="projectStore.keysAsTree()" @item-selected="onItemSelected" />
  </Simplebar>
</template>

<style scoped>
.key-list {
  padding: var(--spacing-padding-large);
  background-color: var(--background-color-elevated-high);
}
</style>
