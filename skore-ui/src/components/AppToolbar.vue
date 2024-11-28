<script setup lang="ts">
import SkoreLogo from "@/components/icons/SkoreLogo.vue";
</script>

<template>
  <div class="app-toolbar">
    <div class="logo">
      <SkoreLogo />
    </div>
    <nav></nav>
  </div>
</template>

<style scoped>
.app-toolbar {
  display: flex;
  width: 60px;
  height: 100dvh;
  flex-direction: column;
  border-right: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  background-color: var(--color-background-secondary);

  & .logo {
    display: flex;
    height: var(--header-height);
    align-items: center;
    justify-content: center;
    border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  }

  & nav {
    flex: 1;
  }
}
</style>
