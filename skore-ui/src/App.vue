<script setup lang="ts">
import { RouterView, useRoute, useRouter } from "vue-router";

import AppToolbar from "@/components/AppToolbar.vue";
import LoadingBars from "@/components/LoadingBars.vue";
import ModalDialog from "@/components/ModalDialog.vue";
import NavigationButton from "@/components/NavigationButton.vue";
import ToastNotificationArea from "@/components/ToastNotificationArea.vue";

const route = useRoute();
const router = useRouter();
</script>

<template>
  <div class="skore">
    <AppToolbar>
      <NavigationButton
        v-for="(r, i) in router.getRoutes()"
        :key="i"
        :icon="`${r.meta['icon']}`"
        :is-selected="r.name == route.name"
        :to="r.path"
      />
    </AppToolbar>
    <RouterView v-slot="{ Component, route }">
      <template v-if="Component">
        <Transition name="fade" mode="out-in">
          <Suspense timeout="0">
            <template #default>
              <component :is="Component" :key="route.path" />
            </template>
            <template #fallback>
              <div class="loader">
                <LoadingBars />
              </div>
            </template>
          </Suspense>
        </Transition>
      </template>
    </RouterView>
  </div>
  <ToastNotificationArea />
  <ModalDialog />
</template>

<style scoped>
.skore {
  display: flex;
  flex-direction: row;

  main {
    width: calc(100dvw - var(--width-toolbar));
    height: 100vh;
  }

  .loader {
    display: flex;
    height: 100dvh;
    flex: 1;
    align-items: center;
    justify-content: center;
  }
}
</style>
