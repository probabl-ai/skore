<script setup lang="ts">
import SimpleButton from "@/components/SimpleButton.vue";
</script>

<template>
  <div class="insert-cell">
    <SimpleButton label="Add a markdown" icon="icon-edit" />
  </div>
</template>

<style scoped>
.insert-cell {
  display: flex;
  align-items: center;
  margin-left: var(--spacing-16);
  gap: var(--spacing-8);

  &::before,
  &::after {
    height: 3px;
    flex: 1;
    border-radius: var(--radius-xs);
    background-color: var(--color-stroke-background-primary);
    content: "";
  }
}
</style>
