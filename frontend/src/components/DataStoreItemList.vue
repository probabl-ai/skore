<script setup lang="ts">
defineProps<{ icon: string; title: string; elements: string[] }>();
</script>

<template>
  <h2><span :class="icon"></span>{{ title }}</h2>
  <div class="elements">
    <div v-for="(element, index) in elements" :key="index" class="element">
      {{ element }}
    </div>
  </div>
</template>

<style scoped>
h2 {
  color: var(--primary-color);
}

.elements {
  display: flex;
  box-sizing: border-box;
  flex: none;
  flex-direction: column;
  flex-grow: 1;
  align-items: flex-start;
  align-self: stretch;
  padding: 16px 24px 24px;
  border-width: 1px 1px 0 0;
  border-style: solid;
  border-color: var(--border-color);
  background: var(--background-color);
  gap: 10px;
}
</style>
