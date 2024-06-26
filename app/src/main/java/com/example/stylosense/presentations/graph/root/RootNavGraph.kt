package com.example.stylosense.presentations.graph.root

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.stylosense.presentations.graph.Graph
import com.example.stylosense.presentations.graph.auth.authNavGraph
import com.example.stylosense.presentations.screen.home_screen.component.HomeScreen
import com.example.stylosense.presentations.screen.profile_screen.ProfileScreen

@Composable
fun RootNavigationGraph(navHostController: NavHostController, context: Context) {
    NavHost(
        navController = navHostController,
        route = Graph.ROOT,
        startDestination = Graph.AUTHENTICATION,
    ) {
        authNavGraph(navHostController, context)
        composable(route = Graph.HOME) {
            HomeScreen()
        }
//        SettingsNavGraph(navHostController, context)
//        composable(route = Graph.SETTINGS) {
//            ProfileScreen(
//                navController = navHostController
//            )
//        }
    }
}