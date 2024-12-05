<script setup lang="ts">
import { computed, onMounted, provide, ref, useSlots } from "vue";

import TabPanelContent from "@/components/TabPanelContent.vue";

const slots = useSlots();
const selectedTabName = ref("");
provide("selectedTabName", selectedTabName);

const tabs = computed(() => {
  const children = slots.default?.() || [];
  return children
    .filter((child) => child.type === TabPanelContent)
    .map((child) => ({
      name: child.props?.name,
      icon: child.props?.icon,
    }));
});

function onTabSelectorClick(tabName: string) {
  selectedTabName.value = tabName;
}

onMounted(() => {
  selectedTabName.value = tabs.value[0].name || "";
});
</script>

<template>
  <div class="tab-panel">
    <div class="tab-selector">
      <div
        v-for="({ name, icon }, i) in tabs"
        :key="i"
        @click="onTabSelectorClick(name)"
        class="tab-selector-element"
        :class="{ selected: name === selectedTabName }"
      >
        <i v-if="icon" class="icon" :class="icon" />
        {{ name }}
      </div>
    </div>
    <div class="tabs">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.tab-panel {
  & .tab-selector {
    display: flex;
    flex-flow: row wrap;
    border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    background-color: var(--color-background-secondary);
    border-top-left-radius: var(--radius-md);
    border-top-right-radius: var(--radius-md);

    & .tab-selector-element {
      padding: var(--spacing-8) var(--spacing-12);
      border-bottom: solid var(--stroke-width-lg) transparent;
      color: var(--color-text-secondary);
      cursor: pointer;
      font-size: var(--font-size-xs);
      transition:
        border-bottom var(--animation-duration) var(--animation-easing),
        color var(--animation-duration) var(--animation-easing);

      & .icon {
        padding-right: var(--spacing-4);
        vertical-align: middle;
      }

      &.selected {
        border-bottom: solid var(--stroke-width-lg) var(--color-icon-tertiary);
        color: var(--color-icon-tertiary);
      }
    }
  }
}
</style>
