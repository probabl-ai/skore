<script setup lang="ts">
import { provide, ref } from "vue";

const props = defineProps<{
  tabNames: string[];
}>();
const currentTabIndex = ref(0);
provide("currentTabIndex", currentTabIndex);

function onTabClick(newTabIndex: number) {
  currentTabIndex.value = newTabIndex;
}
</script>

<template>
  <div class="tabs">
    <div
      v-for="(name, index) in props.tabNames"
      :class="{ selected: currentTabIndex == index }"
      :key="index"
      class="tab"
      @click="onTabClick(index)"
    >
      {{ name }}
    </div>
  </div>
  <slot></slot>
</template>

<style scopes>
.tabs {
  display: flex;
  flex-direction: row;
  border-bottom: 1px solid #e9e9e9;

  & .tab {
    padding: 10px 20px;
    cursor: pointer;
    font-weight: 300;

    &.selected {
      border-bottom: solid 1px black;
      font-weight: 400;
    }
  }
}
</style>
