<script setup lang="ts">
import SkoreLogo from "@/components/icons/SkoreLogo.vue";
</script>

<template>
  <div class="app-toolbar">
    <div class="logo">
      <SkoreLogo />
    </div>
    <nav>
      <slot></slot>
    </nav>
  </div>
</template>

<style scoped>
.app-toolbar {
  display: flex;
  min-width: var(--width-toolbar);
  height: 100dvh;
  flex-direction: column;
  border-right: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  background-color: var(--color-background-secondary);

  & .logo {
    display: flex;
    height: var(--height-header);
    align-items: center;
    justify-content: center;
    border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  }

  & nav {
    display: flex;
    flex: 1;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-12) 0;
    gap: var(--spacing-10);
  }
}
</style>
