<script setup lang="ts">
import { computed, isVNode, onMounted, provide, ref, useSlots, type VNode } from "vue";
import TabPanelContent from "./TabPanelContent.vue";

const slots = useSlots();
const selectedTabName = ref("");
provide("selectedTabName", selectedTabName);

const tabNames = computed(() => {
  const children = slots.default?.() || [];
  return children
    .map((child) => {
      // Ensure it's a VNode with props
      if (isVNode(child)) {
        const node = child as VNode;
        if (node.type === TabPanelContent) {
          return node.props?.name;
        }
      }
      return null;
    })
    .filter((name) => name !== null); // Remove null values
});

function onTabSelectorClick(tabName: string) {
  selectedTabName.value = tabName;
}

onMounted(() => {
  selectedTabName.value = tabNames.value[0] || "";
});
</script>

<template>
  <div class="tab-panel">
    <div class="tab-selector">
      <div
        v-for="(name, i) in tabNames"
        :key="i"
        @click="onTabSelectorClick(name)"
        class="tab-selector-element"
      >
        {{ name }}
      </div>
    </div>
    <div class="tabs">
      <slot />
    </div>
  </div>
</template>
