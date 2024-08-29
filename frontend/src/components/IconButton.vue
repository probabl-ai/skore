<script setup lang="ts">
const props = defineProps<{
  icon: string;
}>();
</script>

<template>
  <button><span :class="props.icon"></span></button>
</template>

<style scoped>
button {
  display: flex;
  width: 24px;
  height: 24px;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  border: var(--border-width-normal) solid var(--border-color-elevated);
  border-radius: var(--border-radius);
  margin: 0 auto;
  background: var(--border-color-lower);
  box-shadow: inset 0 0 3.24px 2.4px rgb(var(--text-color-highlight) 0.5);
  color: var(--text-color-normal);
  cursor: pointer;
  font-size: 14px;
}
</style>
