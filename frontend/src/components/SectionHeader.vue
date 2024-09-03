<script setup lang="ts">
const props = defineProps<{
  title: string;
  icon?: string;
}>();
</script>

<template>
  <div class="header">
    <h1><span v-if="props.icon" class="icon" :class="props.icon"></span>{{ title }}</h1>
  </div>
</template>

<style scoped>
.header {
  display: flex;
  height: 44px;
  align-items: center;
  padding: var(--spacing-padding-large);
  border-right: solid var(--border-width-normal) var(--border-color-normal);
  border-bottom: solid var(--border-width-normal) var(--border-color-normal);
  background-color: var(--background-color-elevated);

  & h1 {
    color: var(--text-color-normal);
    font-size: var(--text-size-title);
    font-weight: var(--text-weight-title);
  }

  & .icon {
    padding-right: 4px;
  }
}
</style>
